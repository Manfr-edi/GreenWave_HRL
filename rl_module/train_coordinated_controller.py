"""Train the full local-controller plus global-coordinator architecture."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from tensordict import TensorDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from rl_module.checkpoints import (
    append_history,
    checkpoint_score_from_baseline,
    checkpoint_score_from_reference,
    load_local_controller_weights,
    resume_coordinated_checkpoint,
    rolling_checkpoint_score,
    write_history_header,
)
from rl_module.coordination import (
    DEFAULT_OBSERVATION_CONFIG,
    DEFAULT_REWARD_CONFIG,
    augment_local_observation_with_prices,
    build_coordination_observation,
    build_taz_adjacency_matrix,
    compute_coordination_step_rewards,
    extract_taz_features,
    summarize_step_actions,
)
from rl_module.environment import CoordinatedTazTrafficEnv
from rl_module.policy import MultiDiscreteActorCritic, PriceAdapterActorCritic
from rl_module.ppo import (
    clone_state_dict_to_cpu,
    compute_gae,
    entropy_coef_now,
    move_optimizer_state_to_device,
    normalize_advantages,
    ppo_update,
    set_global_seed,
)
from rl_module.settings import COORDINATOR, DEMAND, ENVIRONMENT, EXPERIMENT, LOCAL_CONTROLLER
from rl_module.traffic_demand import cleanup_sumo_output_files, build_episode_schedule, ensure_routes, resolve_episode_demand


LOAD_COORDINATED_CHECKPOINT = False
COORDINATED_CHECKPOINT_PATH = os.path.join(EXPERIMENT.checkpoint_dir, "coordinated_controller_best.pt")
RESUME_FROM_CHECKPOINT_EPISODE = False
START_EPISODE_OVERRIDE = None
CHECKPOINT_SCORE_WINDOW = 10
CHECKPOINT_MIN_PROMOTION_EPISODES = 5


def _global_context_tensor(episode, reference_penalty: float | None, device: torch.device) -> torch.Tensor:
    """Compact episode context consumed by the coordinator."""

    reference_value = 0.0
    if COORDINATOR.include_reference_penalty_context and reference_penalty is not None:
        reference_value = float(np.clip(float(reference_penalty), -5.0, 5.0))
    return torch.tensor(
        [
            float(episode.hour) / 23.0,
            float(episode.vehicle_count) / max(float(DEMAND.base_demand), 1.0),
            float(episode.demand_noise),
            reference_value,
        ],
        dtype=torch.float32,
        device=device,
    )


def _build_coordinator_obs_dim(env, adjacency_matrix: torch.Tensor, device: torch.device) -> int:
    """Infer coordinator observation width from the active environment metadata."""

    num_taz = len(env.get_taz_ids())
    dummy_taz_features = torch.zeros((num_taz, env.per_taz_feature_dim), dtype=torch.float32, device=device)
    dummy_flow = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=device)
    dummy_prices = torch.zeros(num_taz, dtype=torch.float32, device=device)
    dummy_context = torch.zeros(4, dtype=torch.float32, device=device)
    return int(
        build_coordination_observation(
            dummy_taz_features,
            adjacency_matrix,
            previous_flow_matrix=dummy_flow,
            previous_prices=dummy_prices,
            taz_ids=env.get_taz_ids(),
            extra_context=dummy_context,
            config=DEFAULT_OBSERVATION_CONFIG,
            price_max=max(COORDINATOR.price_bins),
        ).shape[-1]
    )


def _ensure_baseline_reference(env, sumo, hour: int, route_folder_path: str, cache: dict) -> tuple[float | None, dict, bool]:
    """Run or reuse the no-agent baseline for a specific route folder."""

    key = (route_folder_path, int(hour))
    if key in cache:
        penalty, components = cache[key]
        return penalty, dict(components), False
    sumo.changeRouteFilePath(route_folder_path)
    sumo.changeTypePath(route_folder_path)
    penalty, components = env.run_reference_episode_no_agent(hour=hour)
    if not bool(components.get("parse_ok", False)):
        penalty = None
    cache[key] = (penalty, dict(components))
    return penalty, dict(components), True


def _run_local_reference_episode(
    env,
    sumo,
    local_reference_policy,
    hour: int,
    route_folder_path: str,
    device: torch.device,
    deterministic: bool = True,
) -> tuple[float | None, dict]:
    """Run the frozen local-only controller on the same route/hour."""

    was_training = bool(local_reference_policy.training)
    local_reference_policy.eval()
    sumo.changeRouteFilePath(route_folder_path)
    sumo.changeTypePath(route_folder_path)
    env.set_baseline_penalty(None)
    env.set_episode_context(hour=hour)
    td = env.reset()

    with torch.no_grad():
        while True:
            local_obs = td["observation"]
            action_mask = td["action_mask"].bool()
            action_index, _, _ = local_reference_policy.act(local_obs, action_mask, deterministic=deterministic)
            step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=device))
            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            if terminated or truncated:
                break
            td = next_td

    terminal_components = dict((env.last_reward_components or {}).get("terminal_reward", {}) or {})
    penalty = float(terminal_components.get("penalty", 0.0)) if bool(terminal_components.get("parse_ok", False)) else None
    local_reference_policy.train(mode=was_training)
    return penalty, terminal_components


def _ensure_local_reference(
    env,
    sumo,
    local_reference_policy,
    hour: int,
    route_folder_path: str,
    device: torch.device,
    cache: dict,
    cache_enabled: bool,
) -> tuple[float | None, dict, bool]:
    """Run or reuse the local-only reference for a specific route folder."""

    key = (route_folder_path, int(hour))
    if cache_enabled and key in cache:
        penalty, components = cache[key]
        return penalty, dict(components), False
    penalty, components = _run_local_reference_episode(
        env=env,
        sumo=sumo,
        local_reference_policy=local_reference_policy,
        hour=hour,
        route_folder_path=route_folder_path,
        device=device,
        deterministic=True,
    )
    if cache_enabled:
        cache[key] = (penalty, dict(components))
    return penalty, dict(components), True


def _coordinator_ppo_update(policy, optimizer, rollout: dict, episode_idx: int, total_episodes: int) -> dict:
    """Update the global coordinator with PPO."""

    advantages, returns = compute_gae(
        torch.stack(rollout["rewards"]).reshape(-1, 1),
        torch.cat(rollout["values"], dim=0).reshape(-1, 1),
        torch.stack(rollout["dones"]).reshape(-1, 1),
        gamma=COORDINATOR.gamma,
        lam=COORDINATOR.gae_lambda,
    )
    entropy_coef = entropy_coef_now(
        episode_idx,
        total_episodes,
        COORDINATOR.entropy_coef,
        COORDINATOR.entropy_coef_final,
        LOCAL_CONTROLLER.entropy_warmup_ratio,
    )
    return ppo_update(
        policy,
        optimizer,
        torch.cat(rollout["obs"], dim=0).reshape(-1, rollout["obs_dim"]),
        torch.cat(rollout["action_masks"], dim=0).reshape(-1, rollout["act_dim"]),
        torch.cat(rollout["actions"], dim=0).reshape(-1, rollout["act_dim"]),
        torch.cat(rollout["logp"], dim=0).reshape(-1),
        normalize_advantages(advantages.reshape(-1)),
        returns.reshape(-1),
        clip_ratio=COORDINATOR.clip_ratio,
        ppo_epochs=COORDINATOR.ppo_epochs,
        minibatch_size=COORDINATOR.minibatch_size,
        entropy_coef=entropy_coef,
        value_coef=COORDINATOR.value_coef,
        target_kl=COORDINATOR.target_kl,
        max_grad_norm=LOCAL_CONTROLLER.max_grad_norm,
        joint_action_dim=(
            int(rollout["act_dim"])
            if COORDINATOR.normalize_entropy_and_kl_by_taz
            else 1
        ),
    )


def _save_checkpoint(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _settings_payload() -> dict:
    """Serialize settings with primitive values so torch weights_only loading stays safe."""

    experiment = dict(EXPERIMENT.__dict__)
    experiment["start_date"] = EXPERIMENT.start_date.strftime("%Y-%m-%d")
    return {
        "experiment_settings": experiment,
        "demand_settings": dict(DEMAND.__dict__),
        "environment_settings": dict(ENVIRONMENT.__dict__),
        "local_controller_settings": dict(LOCAL_CONTROLLER.__dict__),
        "coordinator_settings": dict(COORDINATOR.__dict__),
    }


def main():
    """Train the coordinated traffic controller with deterministic randomized demand."""

    parser = argparse.ArgumentParser(description="Train the coordinated traffic controller.")
    parser.add_argument("--dry-run-schedule", action="store_true", help="Print deterministic randomized episodes without running SUMO.")
    parser.add_argument("--show-episodes", type=int, default=10, help="Number of episodes to print with --dry-run-schedule.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for short training/debug runs.")
    args = parser.parse_args()

    set_global_seed(EXPERIMENT.global_seed)
    schedule = build_episode_schedule(EXPERIMENT)
    if args.max_episodes is not None:
        schedule = schedule[: max(int(args.max_episodes), 0)]
    if args.dry_run_schedule:
        for episode_idx, (day, hour) in enumerate(schedule[: max(int(args.show_episodes), 0)]):
            episode = resolve_episode_demand(episode_idx, day, hour, EXPERIMENT, DEMAND)
            print(
                f"{episode_idx}: date={episode.simulation_date} slot={episode.timeslot} "
                f"day_type={episode.day_type} base={episode.base_vehicle_count} "
                f"day_factor={episode.day_type_demand_factor:.4f} noise={episode.demand_noise:.4f} "
                f"vehicles={episode.vehicle_count} path={episode.route_folder_path}"
            )
        print(f"total_episodes={len(schedule)}")
        return

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "command_log_rl_module_coordinated_controller.txt")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

    selected_taz_ids = [str(EXPERIMENT.selected_taz_id)] if EXPERIMENT.selected_taz_id else None
    env = CoordinatedTazTrafficEnv(
        sumoSimulator=sumo,
        stepSize=600,
        selected_taz_ids=selected_taz_ids,
        **ENVIRONMENT.to_env_kwargs(sumo_seed=EXPERIMENT.global_seed),
    )
    device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    num_taz = len(taz_ids)
    base_obs_dim = int(env.agent_obs_dim)
    augmented_obs_dim = base_obs_dim + COORDINATOR.local_price_context_dim
    local_act_dim = int(env.max_control_groups_per_taz)

    adjacency_matrix, adjacency_metadata = build_taz_adjacency_matrix(
        taz_ids=taz_ids,
        taz_additional_path=constants.TAZ_ADDITIONAL_FILE_PATH,
        distance_threshold=COORDINATOR.taz_adjacency_distance_threshold,
        fallback_k=COORDINATOR.taz_adjacency_fallback_k,
        device=device,
    )
    coordinator_obs_dim = _build_coordinator_obs_dim(env, adjacency_matrix, device)

    local_controller = PriceAdapterActorCritic(
        base_obs_dim=base_obs_dim,
        price_context_dim=COORDINATOR.local_price_context_dim,
        act_dim=local_act_dim,
        action_bins=LOCAL_CONTROLLER.action_bins,
        adapter_hidden_dim=COORDINATOR.price_adapter_hidden_dim,
        adapter_scale=COORDINATOR.price_adapter_scale,
        adapter_mode=COORDINATOR.price_adapter_mode,
    ).to(device)
    local_reference_controller = MultiDiscreteActorCritic(base_obs_dim, local_act_dim, LOCAL_CONTROLLER.action_bins).to(device)
    train_local_base_policy = bool(LOCAL_CONTROLLER.train_local_policy)
    if LOCAL_CONTROLLER.load_pretrained:
        # The coordinated stage is normally initialized from the local-only
        # checkpoint. If it is missing, training still works from random weights.
        if os.path.exists(LOCAL_CONTROLLER.pretrained_checkpoint_path):
            pretrained_loaded = load_local_controller_weights(
                local_controller,
                LOCAL_CONTROLLER.pretrained_checkpoint_path,
                base_obs_dim=base_obs_dim,
            )
            reference_loaded = False
            if pretrained_loaded:
                reference_loaded = load_local_controller_weights(
                    local_reference_controller,
                    LOCAL_CONTROLLER.pretrained_checkpoint_path,
                    base_obs_dim=base_obs_dim,
            )
            if not pretrained_loaded:
                print("[WARN] Starting coordinated local controller from random weights.")
                local_reference_controller.load_state_dict(local_controller.base_policy.state_dict())
                if not train_local_base_policy:
                    print("[WARN] No compatible local pretraining was loaded; enabling local base-policy training.")
                    train_local_base_policy = True
            elif not reference_loaded:
                print("[WARN] Local reference controller could not load the checkpoint; using the current base policy.")
                local_reference_controller.load_state_dict(local_controller.base_policy.state_dict())
        else:
            print(f"[WARN] Local pretraining checkpoint not found: {LOCAL_CONTROLLER.pretrained_checkpoint_path}")
            print("[WARN] Starting coordinated local controller from random weights.")
            local_reference_controller.load_state_dict(local_controller.base_policy.state_dict())
            if not train_local_base_policy:
                print("[WARN] No compatible local pretraining was loaded; enabling local base-policy training.")
                train_local_base_policy = True
    else:
        local_reference_controller.load_state_dict(local_controller.base_policy.state_dict())
        if not train_local_base_policy:
            print("[WARN] Local pretraining disabled; enabling local base-policy training.")
            train_local_base_policy = True
    local_reference_controller.eval()
    for parameter in local_reference_controller.parameters():
        parameter.requires_grad_(False)

    local_trainable = bool(train_local_base_policy or LOCAL_CONTROLLER.train_price_adapter)
    local_controller.configure_trainable(
        train_base_policy=train_local_base_policy,
        train_price_adapter=LOCAL_CONTROLLER.train_price_adapter,
    )
    local_controller.train(mode=local_trainable)

    coordinator = MultiDiscreteActorCritic(coordinator_obs_dim, num_taz, COORDINATOR.price_bins).to(device)
    coordinator.train()
    local_optimizer = None
    local_scheduler = None
    local_trainable_parameters = [parameter for parameter in local_controller.parameters() if parameter.requires_grad]
    if local_trainable_parameters:
        local_optimizer = torch.optim.Adam(local_trainable_parameters, lr=LOCAL_CONTROLLER.learning_rate)
        local_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            local_optimizer,
            mode="max",
            factor=LOCAL_CONTROLLER.lr_scheduler_factor,
            patience=LOCAL_CONTROLLER.lr_scheduler_patience,
            threshold=LOCAL_CONTROLLER.lr_scheduler_threshold,
            threshold_mode="rel",
            min_lr=LOCAL_CONTROLLER.min_learning_rate,
        )
    coordinator_optimizer = torch.optim.Adam(coordinator.parameters(), lr=COORDINATOR.learning_rate)
    coordinator_scheduler = None
    if COORDINATOR.use_lr_scheduler:
        coordinator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            coordinator_optimizer,
            mode="max",
            factor=0.7,
            patience=15,
            threshold=0.005,
            threshold_mode="rel",
            min_lr=1e-5,
        )
    start_episode, resumed_best = resume_coordinated_checkpoint(
        checkpoint_path=COORDINATED_CHECKPOINT_PATH,
        enabled=LOAD_COORDINATED_CHECKPOINT,
        resume_episode=RESUME_FROM_CHECKPOINT_EPISODE,
        start_episode_override=START_EPISODE_OVERRIDE,
        local_policy=local_controller,
        coordinator_policy=coordinator,
        local_optimizer=local_optimizer,
        coordinator_optimizer=coordinator_optimizer,
        local_scheduler=local_scheduler,
        coordinator_scheduler=coordinator_scheduler,
    )
    if local_optimizer is not None:
        move_optimizer_state_to_device(local_optimizer, device)
    move_optimizer_state_to_device(coordinator_optimizer, device)

    total_episodes = len(schedule)
    os.makedirs(EXPERIMENT.checkpoint_dir, exist_ok=True)
    os.makedirs(EXPERIMENT.detail_dir, exist_ok=True)
    csv_cols = [
        "episode", "date", "day_type", "timeslot", "hour", "hourly_profile_multiplier",
        "day_type_demand_factor", "base_vehicle_count", "demand_noise", "vehicle_count",
        "terminal_penalty", "baseline_penalty", "delta_penalty_vs_baseline", "local_reference_penalty",
        "delta_penalty_vs_local_reference", "coordination_reference_mode", "coordination_reference_label",
        "coordination_reference_penalty",
        "checkpoint_score", "checkpoint_score_metric", "checkpoint_selection_score", "checkpoint_selection_metric",
        "checkpoint_stability_score", "checkpoint_stability_metric", "checkpoint_selection_window",
        "checkpoint_promotion_qualified", "terminal_parse_ok", "base_local_reward_mean", "shaped_local_reward_mean",
        "coordination_adjustment_mean", "coordinator_reward_mean", "coordinator_price_mean",
        "coordinator_price_std", "mean_emission_penalty", "max_emission_penalty",
        "mean_emission_delta", "total_emission_delta", "worst_taz_worsening",
        "emission_spillover_mean", "emission_imbalance_mean", "spillover_mean", "imbalance_mean", "episode_flow_total",
        "local_ppo_total_loss", "coordinator_ppo_total_loss", "local_learning_rate", "coordinator_learning_rate",
        "random_trip_seed", "route_sampler_seed", "route_folder_path",
    ]
    write_history_header(EXPERIMENT.history_csv_path, csv_cols)
    history = []
    baseline_cache = {}
    local_reference_cache = {}
    best_checkpoint_score = float("inf")
    best_checkpoint_metric = None
    best_terminal_penalty = float("inf")
    best_baseline_penalty = None
    best_local_reference_penalty = None
    best_delta_penalty_vs_baseline = None
    best_delta_penalty_vs_local_reference = None
    best_episode = -1
    best_checkpoint_qualified = False
    if resumed_best:
        if resumed_best["best_checkpoint_score"] is not None:
            best_checkpoint_score = float(resumed_best["best_checkpoint_score"])
        best_checkpoint_metric = resumed_best["best_checkpoint_metric"]
        if resumed_best["best_terminal_penalty"] is not None:
            best_terminal_penalty = float(resumed_best["best_terminal_penalty"])
        best_baseline_penalty = resumed_best["best_baseline_penalty"]
        best_local_reference_penalty = resumed_best["best_local_reference_penalty"]
        best_delta_penalty_vs_baseline = resumed_best["best_delta_penalty_vs_baseline"]
        best_delta_penalty_vs_local_reference = resumed_best["best_delta_penalty_vs_local_reference"]
        best_episode = int(resumed_best["best_episode"])
        best_checkpoint_qualified = bool(resumed_best["best_checkpoint_qualified"])
    last_completed_episode = start_episode - 1
    interrupted = False

    print(
        f"[INFO] coordinated_controller | episodes={total_episodes} | taz={num_taz} | "
        f"base_obs={base_obs_dim} | augmented_obs={augmented_obs_dim} | coordinator_obs={coordinator_obs_dim}"
    )
    print(
        f"[INFO] coordination_reference_mode={COORDINATOR.coordination_reference_mode} | "
        f"train_local_policy={train_local_base_policy} | "
        f"train_price_adapter={LOCAL_CONTROLLER.train_price_adapter}"
    )
    print(f"[INFO] demand_noise_range={DEMAND.demand_noise_range} | schedule_shuffle={EXPERIMENT.shuffle_episode_schedule}")
    print(f"[INFO] route_cache_root={DEMAND.route_cache_root}")

    for episode_idx in range(start_episode, total_episodes):
        # Each episode receives a reproducible but randomized demand level and
        # therefore a distinct route-cache folder.
        day, hour = schedule[episode_idx]
        episode = resolve_episode_demand(episode_idx, day, hour, EXPERIMENT, DEMAND)
        print(
            f"\n[EP {episode_idx + 1}/{total_episodes}] date={episode.simulation_date} slot={episode.timeslot} "
            f"day_type={episode.day_type} vehicles={episode.vehicle_count} "
            f"day_factor={episode.day_type_demand_factor:.4f} noise={episode.demand_noise:.4f}"
        )
        route_folder_path, generated = ensure_routes(episode, planner, DEMAND)
        print(f"[ROUTES] {'generated' if generated else 'cached'} | {route_folder_path}")
        stale_output_cleanup = cleanup_sumo_output_files(route_folder_path)
        if stale_output_cleanup["errors"]:
            print(f"[WARN] SUMO stale output cleanup errors: {stale_output_cleanup['errors'][:3]}")

        baseline_penalty, baseline_components, baseline_ran = _ensure_baseline_reference(
            env=env,
            sumo=sumo,
            hour=episode.hour,
            route_folder_path=route_folder_path,
            cache=baseline_cache,
        )
        print(
            f"[BASELINE] {'computed' if baseline_ran else 'reused'} | "
            f"parse_ok={bool(baseline_components.get('parse_ok', False))} | penalty={baseline_penalty if baseline_penalty is not None else 'n/a'}"
        )
        baseline_cleanup = cleanup_sumo_output_files(route_folder_path)
        if baseline_cleanup["errors"]:
            print(f"[WARN] SUMO baseline output cleanup errors: {baseline_cleanup['errors'][:3]}")

        local_reference_penalty, local_reference_components, local_reference_ran = _ensure_local_reference(
            env=env,
            sumo=sumo,
            local_reference_policy=local_reference_controller,
            hour=episode.hour,
            route_folder_path=route_folder_path,
            device=device,
            cache=local_reference_cache,
            cache_enabled=not bool(LOCAL_CONTROLLER.train_local_policy),
        )
        print(
            f"[LOCAL REF] {'computed' if local_reference_ran else 'reused'} | "
            f"parse_ok={bool(local_reference_components.get('parse_ok', False))} | "
            f"penalty={local_reference_penalty if local_reference_penalty is not None else 'n/a'}"
        )
        local_reference_cleanup = cleanup_sumo_output_files(route_folder_path)
        if local_reference_cleanup["errors"]:
            print(f"[WARN] SUMO local-reference output cleanup errors: {local_reference_cleanup['errors'][:3]}")

        coordination_reference_mode = str(COORDINATOR.coordination_reference_mode)
        if (
            coordination_reference_mode in ("local_only", "local_reference")
            and local_reference_penalty is None
        ):
            print(
                "[WARN] Skipping episode because local-reference mode requires a "
                "valid paired local penalty; falling back to the no-agent baseline "
                "would mix checkpoint objectives."
            )
            continue
        use_local_reference = (
            coordination_reference_mode in ("local_only", "local_reference")
            and local_reference_penalty is not None
        )
        coordination_reference_penalty = local_reference_penalty if use_local_reference else baseline_penalty
        coordination_reference_components = local_reference_components if use_local_reference else baseline_components
        coordination_reference_label = "local_reference" if use_local_reference else "baseline"

        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(route_folder_path)
        if ENVIRONMENT.comparison_reward_enabled and coordination_reference_penalty is not None:
            env.set_baseline_penalty(coordination_reference_penalty, coordination_reference_components)
        else:
            env.set_baseline_penalty(None)
        env.set_episode_context(hour=episode.hour)
        td = env.reset()

        local_rollout = {key: [] for key in ("obs", "action_masks", "actions", "logp", "values", "rewards", "base_rewards", "dones")}
        coordinator_rollout = {key: [] for key in ("obs", "action_masks", "actions", "logp", "values", "rewards", "dones")}
        coordinator_rollout["obs_dim"] = coordinator_obs_dim
        coordinator_rollout["act_dim"] = num_taz
        previous_flow_matrix = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=device)
        previous_prices = torch.zeros(num_taz, dtype=torch.float32, device=device)
        previous_action_summary = None
        step_diagnostics = []
        local_rollout_state = clone_state_dict_to_cpu(local_controller.state_dict())
        coordinator_rollout_state = clone_state_dict_to_cpu(coordinator.state_dict())

        while True:
            # Decision order is important: coordinator prices are generated first,
            # then appended to the local observation before local actions are sampled.
            local_obs = td["observation"]
            action_mask = td["action_mask"].bool()
            coordinator_obs = build_coordination_observation(
                extract_taz_features(local_obs, env),
                adjacency_matrix,
                previous_flow_matrix=previous_flow_matrix,
                previous_prices=previous_prices,
                previous_action_summary=previous_action_summary,
                taz_ids=taz_ids,
                extra_context=_global_context_tensor(episode, coordination_reference_penalty, device),
                config=DEFAULT_OBSERVATION_CONFIG,
                price_max=max(COORDINATOR.price_bins),
            )
            coordinator_action_mask = torch.ones((1, num_taz), dtype=torch.bool, device=device)
            coordinator_action_index, coordinator_logp, coordinator_value = coordinator.act(coordinator_obs, coordinator_action_mask)
            price_values = coordinator.action_values(coordinator_action_index).squeeze(0)
            augmented_local_obs = augment_local_observation_with_prices(
                local_obs,
                price_values,
                adjacency_matrix,
                price_max=max(COORDINATOR.price_bins),
            )
            local_action_index, local_logp, local_value = local_controller.act(augmented_local_obs, action_mask)

            try:
                step_td = env.step(TensorDict({"action": local_action_index}, batch_size=[], device=device))
            except KeyboardInterrupt:
                interrupted = True
                print("[WARN] Interrupted during rollout.")
                break
            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            base_reward_vec = next_td["reward"].detach().clone()
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            done_value = 1.0 if (terminated or truncated) else 0.0
            reward_components = dict(getattr(env, "last_reward_components", {}) or {})
            terminal_reward = dict(reward_components.get("terminal_reward", {}) or {})
            terminal_penalty = float(terminal_reward.get("penalty", 0.0)) if bool(terminal_reward.get("parse_ok", False)) else None

            flow_matrix = torch.tensor(env.get_last_step_flow_matrix(), dtype=torch.float32, device=device)
            current_action_summary = summarize_step_actions(
                taz_ids=taz_ids,
                applied_actions=next_td["applied_action"],
                applied_deltas=next_td["applied_duration_delta"],
                action_mask=action_mask,
            )
            local_adjustment, coordinator_reward_value, coordination_diag = compute_coordination_step_rewards(
                taz_ids=taz_ids,
                price_values=price_values,
                reward_components=reward_components,
                flow_matrix=flow_matrix,
                action_summary_by_taz=current_action_summary,
                baseline_penalty=coordination_reference_penalty if done_value > 0.0 else None,
                terminal_penalty=terminal_penalty if done_value > 0.0 else None,
                config=DEFAULT_REWARD_CONFIG,
                price_max=max(COORDINATOR.price_bins),
            )
            shaped_reward_vec = base_reward_vec
            if local_optimizer is not None:
                shaped_reward_vec = shaped_reward_vec + local_adjustment.to(device=device, dtype=base_reward_vec.dtype)

            local_rollout["obs"].append(augmented_local_obs.detach().clone())
            local_rollout["action_masks"].append(action_mask.detach().clone())
            local_rollout["actions"].append(local_action_index.detach().clone())
            local_rollout["logp"].append(local_logp.detach().clone())
            local_rollout["values"].append(local_value.detach().clone())
            local_rollout["rewards"].append(shaped_reward_vec.detach().clone())
            local_rollout["base_rewards"].append(base_reward_vec.detach().clone())
            local_rollout["dones"].append(torch.full_like(base_reward_vec, done_value))
            coordinator_rollout["obs"].append(coordinator_obs.detach().clone())
            coordinator_rollout["action_masks"].append(coordinator_action_mask.detach().clone())
            coordinator_rollout["actions"].append(coordinator_action_index.detach().clone())
            coordinator_rollout["logp"].append(coordinator_logp.detach().clone())
            coordinator_rollout["values"].append(coordinator_value.detach().clone())
            coordinator_rollout["rewards"].append(torch.tensor([coordinator_reward_value], dtype=torch.float32, device=device))
            coordinator_rollout["dones"].append(torch.tensor([done_value], dtype=torch.float32, device=device))
            step_diagnostics.append(coordination_diag)

            previous_flow_matrix = flow_matrix
            previous_prices = price_values.detach()
            previous_action_summary = current_action_summary
            if terminated or truncated or interrupted:
                break
            td = next_td

        if interrupted:
            break
        if not local_rollout["rewards"]:
            continue

        local_stats = {"total_loss": "", "policy_loss": "", "value_loss": "", "clip_fraction": "", "approx_kl": "", "entropy_mean": ""}
        local_lr = ""
        if local_optimizer is not None:
            local_rewards_t = torch.stack(local_rollout["rewards"])
            local_values_t = torch.stack(local_rollout["values"])
            local_dones_t = torch.stack(local_rollout["dones"])
            local_adv, local_returns = compute_gae(
                local_rewards_t,
                local_values_t,
                local_dones_t,
                gamma=LOCAL_CONTROLLER.gamma,
                lam=LOCAL_CONTROLLER.gae_lambda,
            )
            local_entropy_coef = entropy_coef_now(
                episode_idx,
                total_episodes,
                LOCAL_CONTROLLER.entropy_coef,
                LOCAL_CONTROLLER.entropy_coef_final,
                LOCAL_CONTROLLER.entropy_warmup_ratio,
            )
            local_stats = ppo_update(
                local_controller,
                local_optimizer,
                torch.stack(local_rollout["obs"]).reshape(-1, augmented_obs_dim),
                torch.stack(local_rollout["action_masks"]).reshape(-1, local_act_dim),
                torch.stack(local_rollout["actions"]).reshape(-1, local_act_dim),
                torch.stack(local_rollout["logp"]).reshape(-1),
                normalize_advantages(local_adv.reshape(-1)),
                local_returns.reshape(-1),
                clip_ratio=LOCAL_CONTROLLER.clip_ratio,
                ppo_epochs=LOCAL_CONTROLLER.ppo_epochs,
                minibatch_size=LOCAL_CONTROLLER.minibatch_size,
                entropy_coef=local_entropy_coef,
                value_coef=LOCAL_CONTROLLER.value_coef,
                target_kl=LOCAL_CONTROLLER.target_kl,
                max_grad_norm=LOCAL_CONTROLLER.max_grad_norm,
            )
            local_lr = float(local_optimizer.param_groups[0]["lr"])
            local_scheduler.step(float(torch.stack(local_rollout["rewards"]).sum(dim=0).mean().item()))

        coordinator_stats = _coordinator_ppo_update(coordinator, coordinator_optimizer, coordinator_rollout, episode_idx, total_episodes)
        coordinator_lr = float(coordinator_optimizer.param_groups[0]["lr"])
        coordinator_rewards_t = torch.stack(coordinator_rollout["rewards"]).reshape(-1, 1)
        if coordinator_scheduler is not None:
            coordinator_scheduler.step(float(coordinator_rewards_t.mean().item()))

        terminal_components = dict((env.last_reward_components or {}).get("terminal_reward", {}) or {})
        terminal_parse_ok = bool(terminal_components.get("parse_ok", False))
        terminal_penalty = float(terminal_components.get("penalty", 0.0)) if terminal_parse_ok else None
        _, _, delta_penalty_vs_baseline = checkpoint_score_from_baseline(
            terminal_penalty,
            baseline_penalty,
        )
        delta_penalty_vs_local_reference = None
        if terminal_penalty is not None and local_reference_penalty is not None:
            delta_penalty_vs_local_reference = float(terminal_penalty - local_reference_penalty)
        mean_emission_penalty_score = float(np.mean([diag.get("mean_emission_penalty", 0.0) for diag in step_diagnostics]))
        max_emission_penalty_score = float(np.mean([diag.get("max_emission_penalty", 0.0) for diag in step_diagnostics]))
        mean_emission_delta_score = float(np.mean([diag.get("mean_emission_delta", 0.0) for diag in step_diagnostics]))
        total_emission_delta_score = float(np.mean([diag.get("total_emission_delta", 0.0) for diag in step_diagnostics]))
        worst_taz_worsening_score = float(np.mean([diag.get("worst_taz_worsening", 0.0) for diag in step_diagnostics]))
        emission_spillover_score = float(np.mean([diag.get("emission_spillover", 0.0) for diag in step_diagnostics]))
        emission_imbalance_score = float(np.mean([diag.get("emission_imbalance", 0.0) for diag in step_diagnostics]))
        checkpoint_metric_name = (
            "delta_penalty_vs_local_reference"
            if coordination_reference_label == "local_reference"
            else "delta_penalty_vs_baseline"
        )
        checkpoint_score, checkpoint_metric, _ = checkpoint_score_from_reference(
            terminal_penalty,
            coordination_reference_penalty,
            metric_name=checkpoint_metric_name,
        )
        base_local_reward = torch.stack(local_rollout["base_rewards"]).sum(dim=0)
        shaped_local_reward = torch.stack(local_rollout["rewards"]).sum(dim=0)
        coordinator_prices = torch.cat([coordinator.action_values(actions).reshape(-1).detach().cpu() for actions in coordinator_rollout["actions"]])
        row = {
            "episode": int(episode_idx),
            "date": episode.simulation_date,
            "day_type": episode.day_type,
            "timeslot": episode.timeslot,
            "hour": int(episode.hour),
            "hourly_profile_multiplier": float(episode.hourly_profile_multiplier),
            "day_type_demand_factor": float(episode.day_type_demand_factor),
            "base_vehicle_count": int(episode.base_vehicle_count),
            "demand_noise": float(episode.demand_noise),
            "vehicle_count": int(episode.vehicle_count),
            "terminal_penalty": terminal_penalty if terminal_penalty is not None else "",
            "baseline_penalty": baseline_penalty if baseline_penalty is not None else "",
            "delta_penalty_vs_baseline": (
                delta_penalty_vs_baseline if delta_penalty_vs_baseline is not None else ""
            ),
            "local_reference_penalty": local_reference_penalty if local_reference_penalty is not None else "",
            "delta_penalty_vs_local_reference": (
                delta_penalty_vs_local_reference if delta_penalty_vs_local_reference is not None else ""
            ),
            "coordination_reference_mode": str(COORDINATOR.coordination_reference_mode),
            "coordination_reference_label": coordination_reference_label,
            "coordination_reference_penalty": (
                coordination_reference_penalty if coordination_reference_penalty is not None else ""
            ),
            "checkpoint_score": checkpoint_score if checkpoint_score is not None else "",
            "checkpoint_score_metric": checkpoint_metric or "",
            "terminal_parse_ok": bool(terminal_parse_ok),
            "base_local_reward_mean": float(base_local_reward.mean().item()),
            "shaped_local_reward_mean": float(shaped_local_reward.mean().item()),
            "coordination_adjustment_mean": float((shaped_local_reward - base_local_reward).mean().item()),
            "coordinator_reward_mean": float(coordinator_rewards_t.mean().item()),
            "coordinator_price_mean": float(coordinator_prices.float().mean().item()),
            "coordinator_price_std": float(coordinator_prices.float().std(unbiased=False).item()),
            "mean_emission_penalty": mean_emission_penalty_score,
            "max_emission_penalty": max_emission_penalty_score,
            "mean_emission_delta": mean_emission_delta_score,
            "total_emission_delta": total_emission_delta_score,
            "worst_taz_worsening": worst_taz_worsening_score,
            "emission_spillover_mean": emission_spillover_score,
            "emission_imbalance_mean": emission_imbalance_score,
            "spillover_mean": float(np.mean([diag.get("spillover", 0.0) for diag in step_diagnostics])),
            "imbalance_mean": float(np.mean([diag.get("imbalance", 0.0) for diag in step_diagnostics])),
            "episode_flow_total": float(env.get_episode_flow_matrix().sum()),
            "local_ppo_total_loss": local_stats["total_loss"],
            "coordinator_ppo_total_loss": coordinator_stats["total_loss"],
            "local_learning_rate": local_lr,
            "coordinator_learning_rate": coordinator_lr,
            "random_trip_seed": int(episode.random_trip_seed),
            "route_sampler_seed": int(episode.route_sampler_seed),
            "route_folder_path": route_folder_path,
        }
        checkpoint_stability_score, checkpoint_stability_metric, checkpoint_selection_window = rolling_checkpoint_score(
            [*history, row],
            score_key="checkpoint_score",
            window_size=CHECKPOINT_SCORE_WINDOW,
            min_count=CHECKPOINT_MIN_PROMOTION_EPISODES,
            aggregation="median",
        )
        checkpoint_is_eligible = bool(
            checkpoint_score is not None
            and checkpoint_score <= 0.0
            and checkpoint_stability_score is not None
            and checkpoint_stability_score <= 0.0
        )
        checkpoint_can_seed_best = bool(checkpoint_score is not None and best_episode < 0)
        # The candidate must improve on its own rollout. Rank it by the worse of
        # that current delta and the robust recent trend so one lucky route
        # cannot dominate checkpoint selection. The first valid rollout is kept
        # only as an explicitly provisional fallback for a fresh run.
        checkpoint_selection_score = (
            float(max(checkpoint_score, checkpoint_stability_score))
            if checkpoint_is_eligible
            else float(checkpoint_score)
            if checkpoint_can_seed_best
            else None
        )
        checkpoint_selection_metric = (
            f"max({checkpoint_metric}_current,{checkpoint_stability_metric})"
            if checkpoint_is_eligible
            else f"{checkpoint_metric}_current_provisional"
            if checkpoint_can_seed_best
            else None
        )
        row["checkpoint_selection_score"] = checkpoint_selection_score if checkpoint_selection_score is not None else ""
        row["checkpoint_selection_metric"] = checkpoint_selection_metric or ""
        row["checkpoint_stability_score"] = (
            checkpoint_stability_score
            if checkpoint_stability_score is not None
            else ""
        )
        row["checkpoint_stability_metric"] = checkpoint_stability_metric or ""
        row["checkpoint_selection_window"] = int(checkpoint_selection_window)
        row["checkpoint_promotion_qualified"] = bool(checkpoint_is_eligible)
        append_history(row, EXPERIMENT.history_csv_path, EXPERIMENT.history_json_path, csv_cols, history)

        detail_payload = {
            "episode": int(episode_idx),
            "episode_demand": episode.__dict__,
            "baseline_components": baseline_components,
            "local_reference_components": local_reference_components,
            "coordination_reference_mode": str(COORDINATOR.coordination_reference_mode),
            "coordination_reference_label": coordination_reference_label,
            "coordination_reference_penalty": coordination_reference_penalty,
            "terminal_reward": terminal_components,
            "coordination_step_diagnostics": step_diagnostics,
            "flow_diagnostics": env.get_coordination_flow_diagnostics(),
            "adjacency_metadata": adjacency_metadata,
        }
        with open(os.path.join(EXPERIMENT.detail_dir, f"episode_{episode_idx:04d}.json"), "w", encoding="utf-8") as handle:
            json.dump(detail_payload, handle, indent=2)

        checkpoint_payload = {
            "architecture_name": "coordinated_controller",
            "local_policy_class": "PriceAdapterActorCritic",
            "coordination_reference_mode": str(COORDINATOR.coordination_reference_mode),
            "coordination_reference_label": coordination_reference_label,
            "episode": int(episode_idx),
            "local_controller_state_dict": local_controller.state_dict(),
            "coordinator_state_dict": coordinator.state_dict(),
            "local_optimizer_state_dict": local_optimizer.state_dict() if local_optimizer is not None else None,
            "coordinator_optimizer_state_dict": coordinator_optimizer.state_dict(),
            "local_scheduler_state_dict": local_scheduler.state_dict() if local_scheduler is not None else None,
            "coordinator_scheduler_state_dict": (
                coordinator_scheduler.state_dict()
                if coordinator_scheduler is not None
                else None
            ),
            **_settings_payload(),
            "taz_ids": taz_ids,
            "tls_ids": env.get_tls_ids(),
            "control_groups_by_taz": env.get_control_groups_by_taz(),
            "base_obs_dim": int(base_obs_dim),
            "augmented_obs_dim": int(augmented_obs_dim),
            "coordinator_obs_dim": int(coordinator_obs_dim),
            "coordination_observation_config": DEFAULT_OBSERVATION_CONFIG.__dict__,
            "coordination_reward_config": DEFAULT_REWARD_CONFIG.__dict__,
            "adjacency_metadata": adjacency_metadata,
            "best_episode": int(best_episode),
            "best_checkpoint_metric": best_checkpoint_metric,
            "best_checkpoint_score": best_checkpoint_score if math.isfinite(best_checkpoint_score) else None,
            "best_checkpoint_qualified": bool(best_checkpoint_qualified),
            "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
            "best_baseline_penalty": best_baseline_penalty,
            "best_local_reference_penalty": best_local_reference_penalty,
            "best_delta_penalty_vs_baseline": best_delta_penalty_vs_baseline,
            "best_delta_penalty_vs_local_reference": best_delta_penalty_vs_local_reference,
            "checkpoint_score_metric": checkpoint_metric,
            "checkpoint_score": checkpoint_score,
            "checkpoint_selection_metric": checkpoint_selection_metric,
            "checkpoint_selection_score": checkpoint_selection_score,
            "checkpoint_stability_metric": checkpoint_stability_metric,
            "checkpoint_stability_score": checkpoint_stability_score,
            "checkpoint_selection_window": int(checkpoint_selection_window),
            "checkpoint_promotion_qualified": bool(checkpoint_is_eligible),
            "baseline_penalty": baseline_penalty,
            "delta_penalty_vs_baseline": delta_penalty_vs_baseline,
            "local_reference_penalty": local_reference_penalty,
            "delta_penalty_vs_local_reference": delta_penalty_vs_local_reference,
            "coordination_reference_penalty": coordination_reference_penalty,
        }
        _save_checkpoint(os.path.join(EXPERIMENT.checkpoint_dir, f"coordinated_controller_episode{episode_idx}.pt"), checkpoint_payload)
        should_promote = bool(
            checkpoint_selection_score is not None
            and (
                best_episode < 0
                or (checkpoint_is_eligible and not best_checkpoint_qualified)
                or (
                    checkpoint_is_eligible == best_checkpoint_qualified
                    and checkpoint_selection_score < best_checkpoint_score
                )
            )
        )
        if should_promote:
            best_checkpoint_score = float(checkpoint_selection_score)
            best_checkpoint_metric = checkpoint_selection_metric
            best_checkpoint_qualified = bool(checkpoint_is_eligible)
            best_terminal_penalty = float(terminal_penalty) if terminal_penalty is not None else float("inf")
            best_baseline_penalty = baseline_penalty if baseline_penalty is not None else None
            best_local_reference_penalty = local_reference_penalty if local_reference_penalty is not None else None
            best_delta_penalty_vs_baseline = delta_penalty_vs_baseline
            best_delta_penalty_vs_local_reference = delta_penalty_vs_local_reference
            best_episode = int(episode_idx)
            best_payload = dict(checkpoint_payload)
            best_payload["local_controller_state_dict"] = local_rollout_state
            best_payload["coordinator_state_dict"] = coordinator_rollout_state
            best_payload["local_optimizer_state_dict"] = None
            best_payload["coordinator_optimizer_state_dict"] = None
            best_payload["local_scheduler_state_dict"] = None
            best_payload["coordinator_scheduler_state_dict"] = None
            best_payload["best_episode"] = int(best_episode)
            best_payload["best_checkpoint_metric"] = best_checkpoint_metric
            best_payload["best_checkpoint_score"] = float(best_checkpoint_score)
            best_payload["best_terminal_penalty"] = (
                float(best_terminal_penalty) if math.isfinite(best_terminal_penalty) else None
            )
            best_payload["best_baseline_penalty"] = best_baseline_penalty
            best_payload["best_local_reference_penalty"] = best_local_reference_penalty
            best_payload["best_delta_penalty_vs_baseline"] = best_delta_penalty_vs_baseline
            best_payload["best_delta_penalty_vs_local_reference"] = best_delta_penalty_vs_local_reference
            best_payload["checkpoint_selection_metric"] = checkpoint_selection_metric
            best_payload["checkpoint_selection_score"] = checkpoint_selection_score
            best_payload["checkpoint_stability_metric"] = checkpoint_stability_metric
            best_payload["checkpoint_stability_score"] = checkpoint_stability_score
            best_payload["checkpoint_selection_window"] = int(checkpoint_selection_window)
            best_payload["best_checkpoint_qualified"] = bool(best_checkpoint_qualified)
            best_payload["policy_stage"] = "pre_update_rollout_policy"
            _save_checkpoint(os.path.join(EXPERIMENT.checkpoint_dir, "coordinated_controller_best.pt"), best_payload)
        output_cleanup = cleanup_sumo_output_files(route_folder_path)
        if output_cleanup["errors"]:
            print(f"[WARN] SUMO output cleanup errors: {output_cleanup['errors'][:3]}")
        last_completed_episode = int(episode_idx)
        baseline_text = f"{baseline_penalty:.4f}" if baseline_penalty is not None else "n/a"
        local_ref_text = f"{local_reference_penalty:.4f}" if local_reference_penalty is not None else "n/a"
        delta_baseline_text = f"{delta_penalty_vs_baseline:.4f}" if delta_penalty_vs_baseline is not None else "n/a"
        delta_local_text = (
            f"{delta_penalty_vs_local_reference:.4f}"
            if delta_penalty_vs_local_reference is not None
            else "n/a"
        )
        print(
            f"[RESULT] penalty={terminal_penalty if terminal_penalty is not None else 'n/a'} | "
            f"baseline={baseline_text} | delta_base={delta_baseline_text} | "
            f"local_ref={local_ref_text} | delta_local={delta_local_text} | "
            f"base={row['base_local_reward_mean']:.4f} | shaped={row['shaped_local_reward_mean']:.4f} | "
            f"coord={row['coordinator_reward_mean']:.4f} | price={row['coordinator_price_mean']:.3f} | "
            f"flow={row['episode_flow_total']:.0f}"
        )

    final_payload = {
        "architecture_name": "coordinated_controller",
        "local_policy_class": "PriceAdapterActorCritic",
        "coordination_reference_mode": str(COORDINATOR.coordination_reference_mode),
        "episode": int(last_completed_episode),
        "local_controller_state_dict": local_controller.state_dict(),
        "coordinator_state_dict": coordinator.state_dict(),
        "local_optimizer_state_dict": local_optimizer.state_dict() if local_optimizer is not None else None,
        "coordinator_optimizer_state_dict": coordinator_optimizer.state_dict(),
        "local_scheduler_state_dict": local_scheduler.state_dict() if local_scheduler is not None else None,
        "coordinator_scheduler_state_dict": (
            coordinator_scheduler.state_dict()
            if coordinator_scheduler is not None
            else None
        ),
        **_settings_payload(),
        "taz_ids": taz_ids,
        "tls_ids": env.get_tls_ids(),
        "control_groups_by_taz": env.get_control_groups_by_taz(),
        "base_obs_dim": int(base_obs_dim),
        "augmented_obs_dim": int(augmented_obs_dim),
        "coordinator_obs_dim": int(coordinator_obs_dim),
        "coordination_observation_config": DEFAULT_OBSERVATION_CONFIG.__dict__,
        "coordination_reward_config": DEFAULT_REWARD_CONFIG.__dict__,
        "adjacency_metadata": adjacency_metadata,
        "best_episode": int(best_episode),
        "best_checkpoint_metric": best_checkpoint_metric,
        "best_checkpoint_score": best_checkpoint_score if math.isfinite(best_checkpoint_score) else None,
        "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
        "best_baseline_penalty": best_baseline_penalty,
        "best_local_reference_penalty": best_local_reference_penalty,
        "best_delta_penalty_vs_baseline": best_delta_penalty_vs_baseline,
        "best_delta_penalty_vs_local_reference": best_delta_penalty_vs_local_reference,
        "best_checkpoint_qualified": bool(best_checkpoint_qualified),
        "interrupted": bool(interrupted),
    }
    _save_checkpoint(os.path.join(EXPERIMENT.checkpoint_dir, "coordinated_controller_final.pt"), final_payload)
    try:
        if env.sumo.isLoaded():
            env.sumo.end()
    except Exception:
        pass


if __name__ == "__main__":
    main()

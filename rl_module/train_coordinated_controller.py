from __future__ import annotations

"""Train the full local-controller plus global-coordinator architecture."""

import csv
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
    load_local_controller_weights,
    maybe_resume_coordinated_checkpoint,
    write_history_header,
)
from rl_module.coordination import (
    COORDINATION_PRICE_BINS,
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
from rl_module.policy import MultiDiscreteActorCritic
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
from rl_module.traffic_demand import build_episode_schedule, ensure_routes, resolve_episode_demand


LOAD_COORDINATED_CHECKPOINT = False
COORDINATED_CHECKPOINT_PATH = os.path.join(EXPERIMENT.checkpoint_dir, "coordinated_controller_final.pt")
RESUME_FROM_CHECKPOINT_EPISODE = True
START_EPISODE_OVERRIDE = None


def _global_context_tensor(episode, baseline_penalty: float | None, device: torch.device) -> torch.Tensor:
    """Compact episode context consumed by the coordinator."""

    baseline_value = 0.0 if baseline_penalty is None else float(np.clip(float(baseline_penalty), -5.0, 5.0))
    return torch.tensor(
        [
            float(episode.hour) / 23.0,
            float(episode.vehicle_count) / max(float(DEMAND.base_demand), 1.0),
            float(episode.demand_noise),
            baseline_value,
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
                f"base={episode.base_vehicle_count} noise={episode.demand_noise:.4f} "
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

    local_controller = MultiDiscreteActorCritic(augmented_obs_dim, local_act_dim, LOCAL_CONTROLLER.action_bins).to(device)
    if LOCAL_CONTROLLER.load_pretrained:
        # The coordinated stage is normally initialized from the local-only
        # checkpoint. If it is missing, training still works from random weights.
        if os.path.exists(LOCAL_CONTROLLER.pretrained_checkpoint_path):
            load_local_controller_weights(local_controller, LOCAL_CONTROLLER.pretrained_checkpoint_path, base_obs_dim=base_obs_dim)
        else:
            print(f"[WARN] Local pretraining checkpoint not found: {LOCAL_CONTROLLER.pretrained_checkpoint_path}")
            print("[WARN] Starting coordinated local controller from random weights.")
    local_controller.train(mode=LOCAL_CONTROLLER.train_local_policy)
    if not LOCAL_CONTROLLER.train_local_policy:
        for parameter in local_controller.parameters():
            parameter.requires_grad_(False)

    coordinator = MultiDiscreteActorCritic(coordinator_obs_dim, num_taz, COORDINATOR.price_bins).to(device)
    coordinator.train()
    local_optimizer = None
    local_scheduler = None
    if LOCAL_CONTROLLER.train_local_policy:
        local_optimizer = torch.optim.Adam(local_controller.parameters(), lr=LOCAL_CONTROLLER.learning_rate)
        local_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            local_optimizer, mode="max", factor=0.7, patience=15, threshold=0.005, threshold_mode="rel", min_lr=1e-5
        )
    coordinator_optimizer = torch.optim.Adam(coordinator.parameters(), lr=COORDINATOR.learning_rate)
    coordinator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        coordinator_optimizer, mode="max", factor=0.7, patience=15, threshold=0.005, threshold_mode="rel", min_lr=1e-5
    )
    start_episode = maybe_resume_coordinated_checkpoint(
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
        "episode", "date", "timeslot", "hour", "base_vehicle_count", "demand_noise", "vehicle_count",
        "terminal_penalty", "terminal_parse_ok", "base_local_reward_mean", "shaped_local_reward_mean",
        "coordination_adjustment_mean", "coordinator_reward_mean", "coordinator_price_mean",
        "coordinator_price_std", "spillover_mean", "imbalance_mean", "episode_flow_total",
        "local_ppo_total_loss", "coordinator_ppo_total_loss", "local_learning_rate", "coordinator_learning_rate",
        "random_trip_seed", "route_sampler_seed", "route_folder_path",
    ]
    write_history_header(EXPERIMENT.history_csv_path, csv_cols)
    history = []
    baseline_cache = {}
    best_terminal_penalty = float("inf")
    best_episode = -1
    last_completed_episode = start_episode - 1
    interrupted = False

    print(
        f"[INFO] coordinated_controller | episodes={total_episodes} | taz={num_taz} | "
        f"base_obs={base_obs_dim} | augmented_obs={augmented_obs_dim} | coordinator_obs={coordinator_obs_dim}"
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
            f"vehicles={episode.vehicle_count} noise={episode.demand_noise:.4f}"
        )
        route_folder_path, generated = ensure_routes(episode, planner, DEMAND)
        print(f"[ROUTES] {'generated' if generated else 'cached'} | {route_folder_path}")

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

        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(route_folder_path)
        if ENVIRONMENT.comparison_reward_enabled and baseline_penalty is not None:
            env.set_baseline_penalty(baseline_penalty, baseline_components)
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
                extra_context=_global_context_tensor(episode, baseline_penalty, device),
                config=DEFAULT_OBSERVATION_CONFIG,
            )
            coordinator_action_mask = torch.ones((1, num_taz), dtype=torch.bool, device=device)
            coordinator_action_index, coordinator_logp, coordinator_value = coordinator.act(coordinator_obs, coordinator_action_mask)
            price_values = coordinator.action_values(coordinator_action_index).squeeze(0)
            augmented_local_obs = augment_local_observation_with_prices(local_obs, price_values, adjacency_matrix)
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
                baseline_penalty=baseline_penalty if done_value > 0.0 else None,
                terminal_penalty=terminal_penalty if done_value > 0.0 else None,
                config=DEFAULT_REWARD_CONFIG,
            )
            shaped_reward_vec = base_reward_vec
            if LOCAL_CONTROLLER.train_local_policy:
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
        if LOCAL_CONTROLLER.train_local_policy:
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
        coordinator_scheduler.step(float(coordinator_rewards_t.mean().item()))

        terminal_components = dict((env.last_reward_components or {}).get("terminal_reward", {}) or {})
        terminal_parse_ok = bool(terminal_components.get("parse_ok", False))
        terminal_penalty = float(terminal_components.get("penalty", 0.0)) if terminal_parse_ok else None
        base_local_reward = torch.stack(local_rollout["base_rewards"]).sum(dim=0)
        shaped_local_reward = torch.stack(local_rollout["rewards"]).sum(dim=0)
        coordinator_prices = torch.cat([coordinator.action_values(actions).reshape(-1).detach().cpu() for actions in coordinator_rollout["actions"]])
        row = {
            "episode": int(episode_idx),
            "date": episode.simulation_date,
            "timeslot": episode.timeslot,
            "hour": int(episode.hour),
            "base_vehicle_count": int(episode.base_vehicle_count),
            "demand_noise": float(episode.demand_noise),
            "vehicle_count": int(episode.vehicle_count),
            "terminal_penalty": terminal_penalty if terminal_penalty is not None else "",
            "terminal_parse_ok": bool(terminal_parse_ok),
            "base_local_reward_mean": float(base_local_reward.mean().item()),
            "shaped_local_reward_mean": float(shaped_local_reward.mean().item()),
            "coordination_adjustment_mean": float((shaped_local_reward - base_local_reward).mean().item()),
            "coordinator_reward_mean": float(coordinator_rewards_t.mean().item()),
            "coordinator_price_mean": float(coordinator_prices.float().mean().item()),
            "coordinator_price_std": float(coordinator_prices.float().std(unbiased=False).item()),
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
        append_history(row, EXPERIMENT.history_csv_path, EXPERIMENT.history_json_path, csv_cols, history)

        detail_payload = {
            "episode": int(episode_idx),
            "episode_demand": episode.__dict__,
            "baseline_components": baseline_components,
            "terminal_reward": terminal_components,
            "coordination_step_diagnostics": step_diagnostics,
            "flow_diagnostics": env.get_coordination_flow_diagnostics(),
            "adjacency_metadata": adjacency_metadata,
        }
        with open(os.path.join(EXPERIMENT.detail_dir, f"episode_{episode_idx:04d}.json"), "w", encoding="utf-8") as handle:
            json.dump(detail_payload, handle, indent=2)

        checkpoint_payload = {
            "architecture_name": "coordinated_controller",
            "episode": int(episode_idx),
            "local_controller_state_dict": local_controller.state_dict(),
            "coordinator_state_dict": coordinator.state_dict(),
            "local_optimizer_state_dict": local_optimizer.state_dict() if local_optimizer is not None else None,
            "coordinator_optimizer_state_dict": coordinator_optimizer.state_dict(),
            "local_scheduler_state_dict": local_scheduler.state_dict() if local_scheduler is not None else None,
            "coordinator_scheduler_state_dict": coordinator_scheduler.state_dict(),
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
            "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
        }
        _save_checkpoint(os.path.join(EXPERIMENT.checkpoint_dir, f"coordinated_controller_episode{episode_idx}.pt"), checkpoint_payload)
        if terminal_penalty is not None and terminal_penalty < best_terminal_penalty:
            best_terminal_penalty = float(terminal_penalty)
            best_episode = int(episode_idx)
            best_payload = dict(checkpoint_payload)
            best_payload["local_controller_state_dict"] = local_rollout_state
            best_payload["coordinator_state_dict"] = coordinator_rollout_state
            best_payload["local_optimizer_state_dict"] = None
            best_payload["coordinator_optimizer_state_dict"] = None
            best_payload["local_scheduler_state_dict"] = None
            best_payload["coordinator_scheduler_state_dict"] = None
            best_payload["best_episode"] = int(best_episode)
            best_payload["best_terminal_penalty"] = float(best_terminal_penalty)
            best_payload["policy_stage"] = "pre_update_rollout_policy"
            _save_checkpoint(os.path.join(EXPERIMENT.checkpoint_dir, "coordinated_controller_best.pt"), best_payload)
        last_completed_episode = int(episode_idx)
        print(
            f"[RESULT] penalty={terminal_penalty if terminal_penalty is not None else 'n/a'} | "
            f"base={row['base_local_reward_mean']:.4f} | shaped={row['shaped_local_reward_mean']:.4f} | "
            f"coord={row['coordinator_reward_mean']:.4f} | price={row['coordinator_price_mean']:.3f} | "
            f"flow={row['episode_flow_total']:.0f}"
        )

    final_payload = {
        "architecture_name": "coordinated_controller",
        "episode": int(last_completed_episode),
        "local_controller_state_dict": local_controller.state_dict(),
        "coordinator_state_dict": coordinator.state_dict(),
        "local_optimizer_state_dict": local_optimizer.state_dict() if local_optimizer is not None else None,
        "coordinator_optimizer_state_dict": coordinator_optimizer.state_dict(),
        "local_scheduler_state_dict": local_scheduler.state_dict() if local_scheduler is not None else None,
        "coordinator_scheduler_state_dict": coordinator_scheduler.state_dict(),
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
        "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
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

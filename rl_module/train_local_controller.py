from __future__ import annotations

"""Train only the local TAZ traffic-light controller.

This script uses the RL module components but does not instantiate the
global coordinator. Use it when you want a clean local pretraining stage.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import replace

import torch
from tensordict import TensorDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from rl_module.checkpoints import append_history, load_checkpoint, write_history_header
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
from rl_module.settings import DEMAND, ENVIRONMENT, EXPERIMENT, LOCAL_CONTROLLER, PACKAGE_DIR
from rl_module.traffic_demand import build_episode_schedule, ensure_routes, resolve_episode_demand


# Keep local-only artifacts separate from coordinated-controller artifacts.
LOCAL_EXPERIMENT = replace(EXPERIMENT, name="local_controller")
LOCAL_CHECKPOINT_PATH = os.path.join(LOCAL_EXPERIMENT.checkpoint_dir, "local_controller_final.pt")
LOAD_LOCAL_CHECKPOINT = False
RESUME_FROM_CHECKPOINT_EPISODE = True
START_EPISODE_OVERRIDE = None


def _ensure_baseline_reference(env, sumo, hour: int, route_folder_path: str, cache: dict) -> tuple[float | None, dict, bool]:
    """Run or reuse the no-agent baseline for the current route cache."""

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


def _maybe_load_local_checkpoint(policy, optimizer, scheduler) -> int:
    """Resume a local-controller checkpoint when explicitly enabled."""

    if not LOAD_LOCAL_CHECKPOINT:
        return int(START_EPISODE_OVERRIDE or 0)
    checkpoint = load_checkpoint(LOCAL_CHECKPOINT_PATH)
    state_dict = checkpoint.get("local_controller_state_dict", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise KeyError("Local checkpoint must contain local_controller_state_dict or model_state_dict.")
    policy.load_state_dict(state_dict)
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_episode = int(checkpoint.get("episode", -1)) + 1 if RESUME_FROM_CHECKPOINT_EPISODE else 0
    if START_EPISODE_OVERRIDE is not None:
        start_episode = int(START_EPISODE_OVERRIDE)
    return start_episode


def _save_checkpoint(path: str, payload: dict):
    """Persist model state under the local-controller naming scheme."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _settings_payload() -> dict:
    """Serialize settings using only primitive values for safe checkpoint loading."""

    experiment = dict(LOCAL_EXPERIMENT.__dict__)
    experiment["start_date"] = LOCAL_EXPERIMENT.start_date.strftime("%Y-%m-%d")
    return {
        "experiment_settings": experiment,
        "demand_settings": dict(DEMAND.__dict__),
        "environment_settings": dict(ENVIRONMENT.__dict__),
        "local_controller_settings": dict(LOCAL_CONTROLLER.__dict__),
    }


def main():
    """Run local-only PPO training on randomized deterministic route scenarios."""

    parser = argparse.ArgumentParser(description="Train the local traffic-light controller only.")
    parser.add_argument("--dry-run-schedule", action="store_true", help="Print deterministic randomized episodes without running SUMO.")
    parser.add_argument("--show-episodes", type=int, default=10, help="Number of episodes to print with --dry-run-schedule.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for short training/debug runs.")
    args = parser.parse_args()

    set_global_seed(LOCAL_EXPERIMENT.global_seed)
    schedule = build_episode_schedule(LOCAL_EXPERIMENT)
    if args.max_episodes is not None:
        schedule = schedule[: max(int(args.max_episodes), 0)]
    if args.dry_run_schedule:
        for episode_idx, (day, hour) in enumerate(schedule[: max(int(args.show_episodes), 0)]):
            episode = resolve_episode_demand(episode_idx, day, hour, LOCAL_EXPERIMENT, DEMAND)
            print(
                f"{episode_idx}: date={episode.simulation_date} slot={episode.timeslot} "
                f"base={episode.base_vehicle_count} noise={episode.demand_noise:.4f} "
                f"vehicles={episode.vehicle_count} path={episode.route_folder_path}"
            )
        print(f"total_episodes={len(schedule)}")
        return

    # SUMO setup is intentionally local to this script: the training loop only
    # depends on the stable simulator/planner APIs.
    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "command_log_rl_module_local_controller.txt")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

    selected_taz_ids = [str(LOCAL_EXPERIMENT.selected_taz_id)] if LOCAL_EXPERIMENT.selected_taz_id else None
    env = CoordinatedTazTrafficEnv(
        sumoSimulator=sumo,
        stepSize=600,
        selected_taz_ids=selected_taz_ids,
        **ENVIRONMENT.to_env_kwargs(sumo_seed=LOCAL_EXPERIMENT.global_seed),
    )
    device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    obs_dim = int(env.agent_obs_dim)
    act_dim = int(env.max_control_groups_per_taz)

    policy = MultiDiscreteActorCritic(obs_dim, act_dim, LOCAL_CONTROLLER.action_bins).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LOCAL_CONTROLLER.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.7,
        patience=15,
        threshold=0.005,
        threshold_mode="rel",
        min_lr=1e-5,
    )
    start_episode = _maybe_load_local_checkpoint(policy, optimizer, scheduler)
    move_optimizer_state_to_device(optimizer, device)
    policy.train()

    total_episodes = len(schedule)
    os.makedirs(LOCAL_EXPERIMENT.checkpoint_dir, exist_ok=True)
    os.makedirs(LOCAL_EXPERIMENT.detail_dir, exist_ok=True)
    csv_cols = [
        "episode", "date", "timeslot", "hour", "base_vehicle_count", "demand_noise", "vehicle_count",
        "terminal_penalty", "terminal_parse_ok", "local_reward_mean", "ppo_total_loss", "learning_rate",
        "random_trip_seed", "route_sampler_seed", "route_folder_path",
    ]
    write_history_header(LOCAL_EXPERIMENT.history_csv_path, csv_cols)
    history = []
    baseline_cache = {}
    best_terminal_penalty = float("inf")
    best_episode = -1
    last_completed_episode = start_episode - 1
    interrupted = False

    print(
        f"[INFO] local_controller | episodes={total_episodes} | taz={len(taz_ids)} | "
        f"obs_dim={obs_dim} | act_dim={act_dim} | demand_noise_range={DEMAND.demand_noise_range}"
    )

    for episode_idx in range(start_episode, total_episodes):
        day, hour = schedule[episode_idx]
        episode = resolve_episode_demand(episode_idx, day, hour, LOCAL_EXPERIMENT, DEMAND)
        print(
            f"\n[LOCAL EP {episode_idx + 1}/{total_episodes}] date={episode.simulation_date} "
            f"slot={episode.timeslot} vehicles={episode.vehicle_count} noise={episode.demand_noise:.4f}"
        )
        route_folder_path, generated = ensure_routes(episode, planner, DEMAND)
        print(f"[ROUTES] {'generated' if generated else 'cached'} | {route_folder_path}")

        # Baseline comparison keeps terminal reward scale comparable across randomized demand levels.
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

        obs_list = []
        mask_list = []
        action_list = []
        logp_list = []
        value_list = []
        reward_list = []
        done_list = []
        rollout_state = clone_state_dict_to_cpu(policy.state_dict())

        while True:
            obs = td["observation"]
            action_mask = td["action_mask"].bool()
            action_index, logp, value = policy.act(obs, action_mask)
            try:
                step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=device))
            except KeyboardInterrupt:
                interrupted = True
                print("[WARN] Interrupted during rollout.")
                break
            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            done_value = 1.0 if (terminated or truncated) else 0.0

            obs_list.append(obs.detach().clone())
            mask_list.append(action_mask.detach().clone())
            action_list.append(action_index.detach().clone())
            logp_list.append(logp.detach().clone())
            value_list.append(value.detach().clone())
            reward_list.append(next_td["reward"].detach().clone())
            done_list.append(torch.full_like(next_td["reward"], done_value))

            if terminated or truncated or interrupted:
                break
            td = next_td

        if interrupted:
            break
        if not reward_list:
            continue

        rewards_t = torch.stack(reward_list)
        values_t = torch.stack(value_list)
        dones_t = torch.stack(done_list)
        advantages, returns = compute_gae(
            rewards_t,
            values_t,
            dones_t,
            gamma=LOCAL_CONTROLLER.gamma,
            lam=LOCAL_CONTROLLER.gae_lambda,
        )
        entropy_coef = entropy_coef_now(
            episode_idx,
            total_episodes,
            LOCAL_CONTROLLER.entropy_coef,
            LOCAL_CONTROLLER.entropy_coef_final,
            LOCAL_CONTROLLER.entropy_warmup_ratio,
        )
        stats = ppo_update(
            policy,
            optimizer,
            torch.stack(obs_list).reshape(-1, obs_dim),
            torch.stack(mask_list).reshape(-1, act_dim),
            torch.stack(action_list).reshape(-1, act_dim),
            torch.stack(logp_list).reshape(-1),
            normalize_advantages(advantages.reshape(-1)),
            returns.reshape(-1),
            clip_ratio=LOCAL_CONTROLLER.clip_ratio,
            ppo_epochs=LOCAL_CONTROLLER.ppo_epochs,
            minibatch_size=LOCAL_CONTROLLER.minibatch_size,
            entropy_coef=entropy_coef,
            value_coef=LOCAL_CONTROLLER.value_coef,
            target_kl=LOCAL_CONTROLLER.target_kl,
            max_grad_norm=LOCAL_CONTROLLER.max_grad_norm,
        )
        episode_reward_mean = float(rewards_t.sum(dim=0).mean().item())
        scheduler.step(episode_reward_mean)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        terminal_reward = dict((env.last_reward_components or {}).get("terminal_reward", {}) or {})
        terminal_parse_ok = bool(terminal_reward.get("parse_ok", False))
        terminal_penalty = float(terminal_reward.get("penalty", 0.0)) if terminal_parse_ok else None
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
            "local_reward_mean": episode_reward_mean,
            "ppo_total_loss": stats["total_loss"],
            "learning_rate": learning_rate,
            "random_trip_seed": int(episode.random_trip_seed),
            "route_sampler_seed": int(episode.route_sampler_seed),
            "route_folder_path": route_folder_path,
        }
        append_history(row, LOCAL_EXPERIMENT.history_csv_path, LOCAL_EXPERIMENT.history_json_path, csv_cols, history)

        detail_payload = {
            "episode": int(episode_idx),
            "episode_demand": episode.__dict__,
            "baseline_components": baseline_components,
            "terminal_reward": terminal_reward,
            "reward_components": dict(env.last_reward_components or {}),
            "duration_diagnostics": dict(getattr(env, "last_duration_diagnostics", {}) or {}),
        }
        with open(os.path.join(LOCAL_EXPERIMENT.detail_dir, f"episode_{episode_idx:04d}.json"), "w", encoding="utf-8") as handle:
            json.dump(detail_payload, handle, indent=2)

        checkpoint_payload = {
            "architecture_name": "local_controller",
            "episode": int(episode_idx),
            "local_controller_state_dict": policy.state_dict(),
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            **_settings_payload(),
            "taz_ids": taz_ids,
            "tls_ids": env.get_tls_ids(),
            "control_groups_by_taz": env.get_control_groups_by_taz(),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "best_episode": int(best_episode),
            "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
        }
        _save_checkpoint(os.path.join(LOCAL_EXPERIMENT.checkpoint_dir, f"local_controller_episode{episode_idx}.pt"), checkpoint_payload)
        if terminal_penalty is not None and terminal_penalty < best_terminal_penalty:
            best_terminal_penalty = float(terminal_penalty)
            best_episode = int(episode_idx)
            best_payload = dict(checkpoint_payload)
            best_payload["local_controller_state_dict"] = rollout_state
            best_payload["model_state_dict"] = rollout_state
            best_payload["optimizer_state_dict"] = None
            best_payload["scheduler_state_dict"] = None
            best_payload["best_episode"] = int(best_episode)
            best_payload["best_terminal_penalty"] = float(best_terminal_penalty)
            best_payload["policy_stage"] = "pre_update_rollout_policy"
            _save_checkpoint(os.path.join(LOCAL_EXPERIMENT.checkpoint_dir, "local_controller_best.pt"), best_payload)
        last_completed_episode = int(episode_idx)
        print(
            f"[LOCAL RESULT] penalty={terminal_penalty if terminal_penalty is not None else 'n/a'} | "
            f"reward={episode_reward_mean:.4f} | loss={stats['total_loss']:.4f} | lr={learning_rate:.6f}"
        )

    final_payload = {
        "architecture_name": "local_controller",
        "episode": int(last_completed_episode),
        "local_controller_state_dict": policy.state_dict(),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        **_settings_payload(),
        "taz_ids": taz_ids,
        "tls_ids": env.get_tls_ids(),
        "control_groups_by_taz": env.get_control_groups_by_taz(),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "best_episode": int(best_episode),
        "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
        "interrupted": bool(interrupted),
    }
    _save_checkpoint(os.path.join(LOCAL_EXPERIMENT.checkpoint_dir, "local_controller_final.pt"), final_payload)
    try:
        if env.sumo.isLoaded():
            env.sumo.end()
    except Exception:
        pass


if __name__ == "__main__":
    main()

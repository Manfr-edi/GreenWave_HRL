from __future__ import annotations

"""Evaluate RL controllers against a no-agent baseline.

The report can include the local-only controller, the coordinated controller, or
both. This lets the same route scenario answer two separate questions: whether
local control alone helps, and whether the global coordinator adds value.
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import fields
from datetime import datetime

import numpy as np
import torch
from tensordict import TensorDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from rl_module.coordination import (
    COORDINATION_PRICE_BINS,
    CoordinationObservationConfig,
    CoordinationRewardConfig,
    augment_local_observation_with_prices,
    build_coordination_observation,
    build_taz_adjacency_matrix,
    compute_coordination_step_rewards,
    extract_taz_features,
    summarize_step_actions,
)
from rl_module.environment import CoordinatedTazTrafficEnv
from rl_module.policy import MultiDiscreteActorCritic
from rl_module.settings import COORDINATOR, DEMAND, ENVIRONMENT, EXPERIMENT, LOCAL_CONTROLLER, PACKAGE_DIR
from rl_module.traffic_demand import EpisodeDemand, ensure_routes, resolve_episode_demand


DEFAULT_CHECKPOINT_PATH = os.path.join(PACKAGE_DIR, "checkpoints", "coordinated_controller", "coordinated_controller_best.pt")
DEFAULT_LOCAL_CHECKPOINT_PATH = os.path.join(PACKAGE_DIR, "checkpoints", "local_controller", "local_controller_best.pt")
DEFAULT_REPORT_ROOT = os.path.join(PACKAGE_DIR, "evaluation_reports", "coordinated_controller")
TRIPINFO_EMISSION_ATTRS = {
    "total_co2": "CO2_abs",
    "total_nox": "NOx_abs",
    "total_fuel": "fuel_abs",
    "total_co": "CO_abs",
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if np.isfinite(parsed) else default


def _parse_timeslot(timeslot: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", timeslot.strip())
    if match is None:
        raise ValueError("timeslot must have format HH:MM-HH:MM")
    start_hour, start_minute, end_hour, end_minute = [int(value) for value in match.groups()]
    if start_minute != 0 or end_minute != 0:
        raise ValueError("only full-hour slots are supported")
    if end_hour != (start_hour + 1) % 24:
        raise ValueError("timeslot must span exactly one hour")
    return start_hour, timeslot.replace(":", "-")


def _load_dataclass_config(raw: dict | None, config_cls):
    raw = dict(raw or {})
    valid_keys = {field.name for field in fields(config_cls)}
    return config_cls(**{key: raw[key] for key in raw if key in valid_keys})


def _parse_tripinfo_emission_totals(tripinfo_path: str) -> dict:
    totals = {metric_name: 0.0 for metric_name in TRIPINFO_EMISSION_ATTRS}
    try:
        root = ET.parse(tripinfo_path).getroot()
    except Exception:
        return {**totals, "tripinfo_count": 0, "emission_trip_count": 0, "ok": False, "path": tripinfo_path}

    seen_metrics = {metric_name: False for metric_name in TRIPINFO_EMISSION_ATTRS}
    tripinfo_count = 0
    emission_trip_count = 0
    for trip in root.findall("tripinfo"):
        tripinfo_count += 1
        emissions = trip.find("emissions")
        if emissions is None:
            continue
        saw_emission_value = False
        for metric_name, attr_name in TRIPINFO_EMISSION_ATTRS.items():
            value = emissions.get(attr_name)
            if value is None:
                continue
            totals[metric_name] += _safe_float(value, 0.0)
            seen_metrics[metric_name] = True
            saw_emission_value = True
        if saw_emission_value:
            emission_trip_count += 1
    return {
        **{metric_name: float(value) for metric_name, value in totals.items()},
        "tripinfo_count": int(tripinfo_count),
        "emission_trip_count": int(emission_trip_count),
        "ok": bool(emission_trip_count > 0 and all(seen_metrics.values())),
        "path": tripinfo_path,
    }


def _collect_run_metrics(run_root: str, waiting_components: dict | None = None) -> dict:
    output_dir = os.path.join(run_root, "output")
    waiting = dict(waiting_components or {})
    tripinfo_path = str(waiting.get("tripinfo_path") or os.path.join(output_dir, "tripinfos.xml"))
    emissions = _parse_tripinfo_emission_totals(tripinfo_path)
    return {
        "trip_count": int(waiting.get("trip_count", 0)),
        "tripinfo_count": int(emissions["tripinfo_count"]),
        "emission_trip_count": int(emissions["emission_trip_count"]),
        "avg_waiting_time": float(waiting.get("avg_waiting_time", 0.0)),
        "total_waiting_time": float(waiting.get("total_waiting_time", 0.0)),
        "waiting_metric_source": str(waiting.get("waiting_metric_source", "unknown")),
        "emission_metric_source": "tripinfo_emissions_abs",
        "waiting_time_by_taz": dict(waiting.get("waiting_time_by_taz", {}) or {}),
        "vehicle_count_by_taz": dict(waiting.get("vehicle_count_by_taz", {}) or {}),
        "total_co2": float(emissions["total_co2"]),
        "total_nox": float(emissions["total_nox"]),
        "total_fuel": float(emissions["total_fuel"]),
        "total_co": float(emissions["total_co"]),
        "ok": bool(waiting.get("parse_ok", False) and emissions["ok"]),
        "tripinfo_path": tripinfo_path,
        "emission_path": tripinfo_path,
        "emission_output_path": os.path.join(output_dir, "emission-output.xml"),
    }


def _evaluation_episode(simulation_date: str, hour: int, episode_index: int, demand_noise_override: float | None) -> EpisodeDemand:
    day = datetime.strptime(simulation_date, "%Y-%m-%d")
    episode = resolve_episode_demand(episode_index, day, hour, EXPERIMENT, DEMAND)
    if demand_noise_override is None:
        return episode
    vehicle_count = max(1, int(round(episode.base_vehicle_count * float(demand_noise_override))))
    return EpisodeDemand(
        episode_idx=int(episode_index),
        simulation_date=episode.simulation_date,
        hour=episode.hour,
        timeslot=episode.timeslot,
        timeslot_clean=episode.timeslot_clean,
        base_vehicle_count=episode.base_vehicle_count,
        demand_noise=float(demand_noise_override),
        vehicle_count=vehicle_count,
        random_trip_seed=episode.random_trip_seed,
        route_sampler_seed=episode.route_sampler_seed,
        route_folder_path=episode.route_folder_path.replace(f"demand_{episode.vehicle_count}", f"demand_{vehicle_count}").replace(
            f"noise_{episode.demand_noise:.4f}".replace(".", "p"),
            f"noise_{float(demand_noise_override):.4f}".replace(".", "p"),
        ),
    )


def _global_context_tensor(episode: EpisodeDemand, baseline_penalty: float | None, device: torch.device) -> torch.Tensor:
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


def _build_policy_bundle(checkpoint: dict, env: CoordinatedTazTrafficEnv, adjacency_matrix: torch.Tensor) -> dict:
    device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    num_taz = len(taz_ids)
    obs_config = _load_dataclass_config(checkpoint.get("coordination_observation_config"), CoordinationObservationConfig)
    reward_config = _load_dataclass_config(checkpoint.get("coordination_reward_config"), CoordinationRewardConfig)
    price_bins = tuple(checkpoint.get("coordinator_settings", {}).get("price_bins", checkpoint.get("coordination_price_bins", COORDINATION_PRICE_BINS)))
    local_action_bins = tuple(checkpoint.get("local_controller_settings", {}).get("action_bins", LOCAL_CONTROLLER.action_bins))
    dummy_taz_features = torch.zeros((num_taz, env.per_taz_feature_dim), dtype=torch.float32, device=device)
    dummy_flow = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=device)
    dummy_prices = torch.zeros(num_taz, dtype=torch.float32, device=device)
    dummy_context = torch.zeros(4, dtype=torch.float32, device=device)
    coordinator_obs_dim = int(
        build_coordination_observation(
            dummy_taz_features,
            adjacency_matrix,
            previous_flow_matrix=dummy_flow,
            previous_prices=dummy_prices,
            taz_ids=taz_ids,
            extra_context=dummy_context,
            config=obs_config,
        ).shape[-1]
    )
    local_obs_dim = int(checkpoint.get("augmented_obs_dim", env.agent_obs_dim + COORDINATOR.local_price_context_dim))
    local_policy = MultiDiscreteActorCritic(local_obs_dim, env.max_control_groups_per_taz, local_action_bins).to(device)
    coordinator = MultiDiscreteActorCritic(coordinator_obs_dim, num_taz, price_bins).to(device)
    local_state = checkpoint.get("local_controller_state_dict", checkpoint.get("local_model_state_dict"))
    coordinator_state = checkpoint.get("coordinator_state_dict", checkpoint.get("global_model_state_dict"))
    if local_state is None or coordinator_state is None:
        raise KeyError("Checkpoint must contain local_controller_state_dict and coordinator_state_dict.")
    local_policy.load_state_dict(local_state)
    coordinator.load_state_dict(coordinator_state)
    local_policy.eval()
    coordinator.eval()
    return {
        "local_policy": local_policy,
        "coordinator": coordinator,
        "observation_config": obs_config,
        "reward_config": reward_config,
        "price_bins": price_bins,
        "local_action_bins": local_action_bins,
        "coordinator_obs_dim": int(coordinator_obs_dim),
        "local_obs_dim": int(local_obs_dim),
    }


def _build_local_policy_bundle(checkpoint: dict, env: CoordinatedTazTrafficEnv) -> dict:
    """Load a local-only controller checkpoint without coordinator price features."""

    device = torch.device(env.device)
    local_action_bins = tuple(checkpoint.get("local_controller_settings", {}).get("action_bins", LOCAL_CONTROLLER.action_bins))
    obs_dim = int(checkpoint.get("obs_dim", env.agent_obs_dim))
    act_dim = int(checkpoint.get("act_dim", env.max_control_groups_per_taz))
    local_policy = MultiDiscreteActorCritic(obs_dim, act_dim, local_action_bins).to(device)
    local_state = checkpoint.get("local_controller_state_dict", checkpoint.get("model_state_dict", checkpoint.get("local_model_state_dict")))
    if local_state is None:
        raise KeyError("Local checkpoint must contain local_controller_state_dict or model_state_dict.")
    local_policy.load_state_dict(local_state)
    local_policy.eval()
    return {
        "local_policy": local_policy,
        "local_action_bins": local_action_bins,
        "local_obs_dim": int(obs_dim),
        "local_act_dim": int(act_dim),
    }


def _matrix_to_nested_dict(matrix: torch.Tensor, taz_ids: list[str]) -> dict:
    matrix_cpu = matrix.detach().cpu()
    return {
        src_taz: {
            dst_taz: float(matrix_cpu[src_idx, dst_idx].item())
            for dst_idx, dst_taz in enumerate(taz_ids)
            if float(matrix_cpu[src_idx, dst_idx].item()) > 0.0
        }
        for src_idx, src_taz in enumerate(taz_ids)
    }


def _run_local_episode(env, bundle: dict, episode: EpisodeDemand, deterministic: bool) -> dict:
    """Run one episode with the local controller only."""

    device = torch.device(env.device)
    action_by_step = []
    flow_by_step = []
    env.set_baseline_penalty(None)
    env.set_episode_context(hour=episode.hour)
    td = env.reset()

    with torch.no_grad():
        while True:
            local_obs = td["observation"]
            action_mask = td["action_mask"].bool()
            action_index, _, _ = bundle["local_policy"].act(local_obs, action_mask, deterministic=deterministic)
            step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=device))
            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            action_values = bundle["local_policy"].action_values(action_index)
            action_by_step.append({
                taz: [
                    float(action_values[taz_idx, group_idx].item())
                    for group_idx in range(action_values.shape[1])
                    if bool(action_mask[taz_idx, group_idx].item())
                ]
                for taz_idx, taz in enumerate(env.get_taz_ids())
            })
            flow_by_step.append(
                _matrix_to_nested_dict(
                    torch.tensor(env.get_last_step_flow_matrix(), dtype=torch.float32, device=device),
                    env.get_taz_ids(),
                )
            )
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            if terminated or truncated:
                break
            td = next_td

    return {
        "terminal_reward": dict((env.last_reward_components or {}).get("terminal_reward", {}) or {}),
        "group_action_details": dict((env.last_reward_components or {}).get("group_action_details", {}) or {}),
        "group_signal_by_taz": dict((env.last_reward_components or {}).get("group_signal_by_taz", {}) or {}),
        "local_rollout": {
            "enabled": True,
            "action_by_step": action_by_step,
            "flow_by_step": flow_by_step,
            "episode_flow_by_taz": env.get_coordination_flow_diagnostics().get("episode_flow_by_taz", {}),
        },
    }


def _run_rl_episode(env, bundle: dict, adjacency_matrix: torch.Tensor, episode: EpisodeDemand, deterministic: bool, baseline_penalty: float | None) -> dict:
    device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    num_taz = len(taz_ids)
    previous_flow_matrix = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=device)
    previous_prices = torch.zeros(num_taz, dtype=torch.float32, device=device)
    previous_action_summary = None
    step_prices = []
    step_diagnostics = []
    step_flows = []
    env.set_baseline_penalty(None)
    env.set_episode_context(hour=episode.hour)
    td = env.reset()

    with torch.no_grad():
        while True:
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
                config=bundle["observation_config"],
            )
            coordinator_action_mask = torch.ones((1, num_taz), dtype=torch.bool, device=device)
            coordinator_action_index, _, _ = bundle["coordinator"].act(coordinator_obs, coordinator_action_mask, deterministic=deterministic)
            price_values = bundle["coordinator"].action_values(coordinator_action_index).squeeze(0)
            augmented_local_obs = augment_local_observation_with_prices(local_obs, price_values, adjacency_matrix)
            local_action_index, _, _ = bundle["local_policy"].act(augmented_local_obs, action_mask, deterministic=deterministic)
            step_td = env.step(TensorDict({"action": local_action_index}, batch_size=[], device=device))
            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            reward_components = dict(getattr(env, "last_reward_components", {}) or {})
            terminal_reward = dict(reward_components.get("terminal_reward", {}) or {})
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            done = terminated or truncated
            terminal_penalty = float(terminal_reward.get("penalty", 0.0)) if done and bool(terminal_reward.get("parse_ok", False)) else None
            flow_matrix = torch.tensor(env.get_last_step_flow_matrix(), dtype=torch.float32, device=device)
            action_summary = summarize_step_actions(
                taz_ids=taz_ids,
                applied_actions=next_td["applied_action"],
                applied_deltas=next_td["applied_duration_delta"],
                action_mask=action_mask,
            )
            _, coordinator_reward, diag = compute_coordination_step_rewards(
                taz_ids=taz_ids,
                price_values=price_values,
                reward_components=reward_components,
                flow_matrix=flow_matrix,
                action_summary_by_taz=action_summary,
                baseline_penalty=baseline_penalty if done else None,
                terminal_penalty=terminal_penalty if done else None,
                config=bundle["reward_config"],
            )
            diag["coordinator_reward"] = float(coordinator_reward)
            step_prices.append({taz: float(price_values[idx].item()) for idx, taz in enumerate(taz_ids)})
            step_diagnostics.append(diag)
            step_flows.append(_matrix_to_nested_dict(flow_matrix, taz_ids))
            previous_flow_matrix = flow_matrix
            previous_prices = price_values.detach()
            previous_action_summary = action_summary
            if done:
                break
            td = next_td

    return {
        "terminal_reward": dict((env.last_reward_components or {}).get("terminal_reward", {}) or {}),
        "group_action_details": dict((env.last_reward_components or {}).get("group_action_details", {}) or {}),
        "group_signal_by_taz": dict((env.last_reward_components or {}).get("group_signal_by_taz", {}) or {}),
        "coordination_rollout": {
            "enabled": True,
            "price_by_step": step_prices,
            "flow_by_step": step_flows,
            "step_diagnostics": step_diagnostics,
            "episode_flow_by_taz": env.get_coordination_flow_diagnostics().get("episode_flow_by_taz", {}),
        },
    }


def _metric_comparison(baseline_value: float, rl_value: float) -> dict:
    delta = float(rl_value - baseline_value)
    improvement = float(baseline_value - rl_value)
    improvement_pct = float(improvement / baseline_value * 100.0) if abs(baseline_value) > 1e-9 else None
    return {"baseline": float(baseline_value), "rl": float(rl_value), "delta_rl_minus_baseline": delta, "improvement_vs_baseline": improvement, "improvement_pct": improvement_pct}


def _build_comparison(baseline_metrics: dict, rl_metrics: dict) -> dict:
    return {
        metric: _metric_comparison(baseline_metrics[metric], rl_metrics[metric])
        for metric in ("avg_waiting_time", "total_waiting_time", "total_co2", "total_nox", "total_fuel", "total_co")
    }


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    def _mean(key: str) -> float:
        return float(np.mean([_safe_float(metrics.get(key), 0.0) for metrics in metrics_list]))

    return {
        "run_count": int(len(metrics_list)),
        "trip_count": _mean("trip_count"),
        "tripinfo_count": _mean("tripinfo_count"),
        "emission_trip_count": _mean("emission_trip_count"),
        "avg_waiting_time": _mean("avg_waiting_time"),
        "total_waiting_time": _mean("total_waiting_time"),
        "total_co2": _mean("total_co2"),
        "total_nox": _mean("total_nox"),
        "total_fuel": _mean("total_fuel"),
        "total_co": _mean("total_co"),
        "ok": bool(all(bool(metrics.get("ok", False)) for metrics in metrics_list)),
        "run_dirs": [str(metrics.get("run_dir", "")) for metrics in metrics_list],
    }


def _print_summary(report: dict):
    section_names = [
        "comparison_local_greedy",
        "comparison_local_stochastic_mean",
        "comparison_coordinated_greedy",
        "comparison_coordinated_stochastic_mean",
    ]
    for section_name in section_names:
        if section_name not in report:
            continue
        print(section_name)
        for metric_name, values in report[section_name].items():
            pct = values.get("improvement_pct")
            pct_text = "n/a" if pct is None else f"{pct:.2f}%"
            print(
                f"{metric_name}: baseline={values['baseline']:.4f} | rl={values['rl']:.4f} | "
                f"delta={values['delta_rl_minus_baseline']:.4f} | improvement={values['improvement_vs_baseline']:.4f} | improvement_pct={pct_text}"
            )
        print("")


def main():
    parser = argparse.ArgumentParser(description="Compare no-agent baseline against RL controllers.")
    parser.add_argument("--date", required=True, help="Simulation date in YYYY-MM-DD format.")
    parser.add_argument("--timeslot", required=True, help="Timeslot in HH:MM-HH:MM format.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Path to coordinated_controller checkpoint.")
    parser.add_argument("--local-checkpoint", default=DEFAULT_LOCAL_CHECKPOINT_PATH, help="Path to local_controller checkpoint.")
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT, help="Root folder for evaluation outputs.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index used to reproduce demand noise for this date/hour.")
    parser.add_argument("--demand-noise", type=float, default=None, help="Optional explicit demand multiplier.")
    parser.add_argument("--stochastic-runs", type=int, default=1, help="Sampled-policy runs to average.")
    parser.add_argument("--skip-local", action="store_true", help="Skip local-only controller evaluation.")
    parser.add_argument("--skip-coordinated", action="store_true", help="Skip coordinated-controller evaluation.")
    args = parser.parse_args()

    if args.stochastic_runs < 1:
        raise ValueError("--stochastic-runs must be >= 1")
    run_local = bool(not args.skip_local and args.local_checkpoint and os.path.exists(args.local_checkpoint))
    run_coordinated = bool(not args.skip_coordinated and args.checkpoint and os.path.exists(args.checkpoint))
    if not args.skip_local and not run_local:
        print(f"[WARN] Local checkpoint not found, skipping local-only evaluation: {args.local_checkpoint}")
    if not args.skip_coordinated and not run_coordinated:
        print(f"[WARN] Coordinated checkpoint not found, skipping coordinated evaluation: {args.checkpoint}")
    if not run_local and not run_coordinated:
        raise FileNotFoundError("No usable RL checkpoint found. Provide --local-checkpoint and/or --checkpoint.")

    hour, timeslot_clean = _parse_timeslot(args.timeslot)
    episode = _evaluation_episode(args.date, hour, args.episode_index, args.demand_noise)
    local_checkpoint = torch.load(args.local_checkpoint, map_location="cpu", weights_only=True) if run_local else None
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if run_coordinated else None
    checkpoint_taz_ids = []
    if checkpoint is not None:
        checkpoint_taz_ids = [str(taz) for taz in checkpoint.get("taz_ids", [])]
    elif local_checkpoint is not None:
        checkpoint_taz_ids = [str(taz) for taz in local_checkpoint.get("taz_ids", [])]
    selected_taz_ids = checkpoint_taz_ids if len(checkpoint_taz_ids) == 1 else None
    report_dir = os.path.join(os.path.abspath(args.report_root), args.date, timeslot_clean, f"demand_{episode.vehicle_count}")
    baseline_run_dir = os.path.join(report_dir, "baseline")
    local_greedy_run_dir = os.path.join(report_dir, "local_greedy")
    local_stochastic_root_dir = os.path.join(report_dir, "local_stochastic")
    coordinated_greedy_run_dir = os.path.join(report_dir, "coordinated_greedy")
    coordinated_stochastic_root_dir = os.path.join(report_dir, "coordinated_stochastic")
    os.makedirs(os.path.join(baseline_run_dir, "output"), exist_ok=True)
    if run_local:
        os.makedirs(os.path.join(local_greedy_run_dir, "output"), exist_ok=True)
        os.makedirs(local_stochastic_root_dir, exist_ok=True)
    if run_coordinated:
        os.makedirs(os.path.join(coordinated_greedy_run_dir, "output"), exist_ok=True)
        os.makedirs(coordinated_stochastic_root_dir, exist_ok=True)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "compare_rl_module_coordinated_controller.log")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)
    env = CoordinatedTazTrafficEnv(
        sumoSimulator=sumo,
        stepSize=600,
        selected_taz_ids=selected_taz_ids,
        **ENVIRONMENT.to_env_kwargs(sumo_seed=EXPERIMENT.global_seed),
    )

    try:
        route_folder_path, generated = ensure_routes(episode, planner, DEMAND)
        print(f"[ROUTES] {'generated' if generated else 'cached'} | {route_folder_path}")
        adjacency_matrix, adjacency_metadata = build_taz_adjacency_matrix(
            taz_ids=env.get_taz_ids(),
            taz_additional_path=constants.TAZ_ADDITIONAL_FILE_PATH,
            distance_threshold=COORDINATOR.taz_adjacency_distance_threshold,
            fallback_k=COORDINATOR.taz_adjacency_fallback_k,
            device=torch.device(env.device),
        )
        local_bundle = _build_local_policy_bundle(local_checkpoint, env) if run_local else None
        coordinated_bundle = _build_policy_bundle(checkpoint, env, adjacency_matrix) if run_coordinated else None

        print("[BASELINE] Running no-agent baseline.")
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(baseline_run_dir)
        _, baseline_components = env.run_reference_episode_no_agent(hour=episode.hour)
        baseline_metrics = _collect_run_metrics(baseline_run_dir, waiting_components=baseline_components)
        baseline_metrics["run_dir"] = baseline_run_dir
        baseline_penalty = float(baseline_components.get("penalty", 0.0)) if baseline_components.get("parse_ok", False) else None

        local_greedy_metrics = None
        local_stochastic_runs = []
        local_stochastic_mean = None
        if run_local:
            print("[LOCAL GREEDY] Running deterministic local-only controller.")
            sumo.changeRouteFilePath(route_folder_path)
            sumo.changeTypePath(local_greedy_run_dir)
            rollout = _run_local_episode(env, local_bundle, episode, deterministic=True)
            local_greedy_metrics = _collect_run_metrics(local_greedy_run_dir, waiting_components=rollout["terminal_reward"])
            local_greedy_metrics["run_dir"] = local_greedy_run_dir
            local_greedy_metrics["local_rollout"] = rollout["local_rollout"]
            local_greedy_metrics["group_action_details"] = rollout["group_action_details"]
            local_greedy_metrics["group_signal_by_taz"] = rollout["group_signal_by_taz"]

            for run_idx in range(args.stochastic_runs):
                run_dir = os.path.join(local_stochastic_root_dir, f"run_{run_idx + 1:02d}")
                os.makedirs(os.path.join(run_dir, "output"), exist_ok=True)
                print(f"[LOCAL STOCHASTIC] Run {run_idx + 1}/{args.stochastic_runs}.")
                sumo.changeRouteFilePath(route_folder_path)
                sumo.changeTypePath(run_dir)
                rollout = _run_local_episode(env, local_bundle, episode, deterministic=False)
                metrics = _collect_run_metrics(run_dir, waiting_components=rollout["terminal_reward"])
                metrics["run_dir"] = run_dir
                metrics["run_index"] = int(run_idx + 1)
                metrics["local_rollout"] = rollout["local_rollout"]
                local_stochastic_runs.append(metrics)
            local_stochastic_mean = _aggregate_metrics(local_stochastic_runs)

        coordinated_greedy_metrics = None
        coordinated_stochastic_runs = []
        coordinated_stochastic_mean = None
        if run_coordinated:
            print("[COORDINATED GREEDY] Running deterministic coordinated controller.")
            sumo.changeRouteFilePath(route_folder_path)
            sumo.changeTypePath(coordinated_greedy_run_dir)
            rollout = _run_rl_episode(env, coordinated_bundle, adjacency_matrix, episode, deterministic=True, baseline_penalty=baseline_penalty)
            coordinated_greedy_metrics = _collect_run_metrics(coordinated_greedy_run_dir, waiting_components=rollout["terminal_reward"])
            coordinated_greedy_metrics["run_dir"] = coordinated_greedy_run_dir
            coordinated_greedy_metrics["coordination_rollout"] = rollout["coordination_rollout"]
            coordinated_greedy_metrics["group_action_details"] = rollout["group_action_details"]
            coordinated_greedy_metrics["group_signal_by_taz"] = rollout["group_signal_by_taz"]

            for run_idx in range(args.stochastic_runs):
                run_dir = os.path.join(coordinated_stochastic_root_dir, f"run_{run_idx + 1:02d}")
                os.makedirs(os.path.join(run_dir, "output"), exist_ok=True)
                print(f"[COORDINATED STOCHASTIC] Run {run_idx + 1}/{args.stochastic_runs}.")
                sumo.changeRouteFilePath(route_folder_path)
                sumo.changeTypePath(run_dir)
                rollout = _run_rl_episode(env, coordinated_bundle, adjacency_matrix, episode, deterministic=False, baseline_penalty=baseline_penalty)
                metrics = _collect_run_metrics(run_dir, waiting_components=rollout["terminal_reward"])
                metrics["run_dir"] = run_dir
                metrics["run_index"] = int(run_idx + 1)
                metrics["coordination_rollout"] = rollout["coordination_rollout"]
                coordinated_stochastic_runs.append(metrics)
            coordinated_stochastic_mean = _aggregate_metrics(coordinated_stochastic_runs)

        report = {
            "date": args.date,
            "timeslot": args.timeslot,
            "episode_index": int(args.episode_index),
            "episode_demand": episode.__dict__,
            "local_checkpoint_path": os.path.abspath(args.local_checkpoint) if run_local else None,
            "local_checkpoint_episode": int(local_checkpoint.get("episode", -1)) if run_local else None,
            "coordinated_checkpoint_path": os.path.abspath(args.checkpoint) if run_coordinated else None,
            "coordinated_checkpoint_episode": int(checkpoint.get("episode", -1)) if run_coordinated else None,
            "architecture_name": "baseline_vs_local_vs_coordinated",
            "route_folder_path": route_folder_path,
            "route_generated": bool(generated),
            "taz_ids": env.get_taz_ids(),
            "tls_ids": env.get_tls_ids(),
            "coordination_price_bins": list(coordinated_bundle["price_bins"]) if run_coordinated else None,
            "adjacency_metadata": adjacency_metadata,
            "coordinator_obs_dim": int(coordinated_bundle["coordinator_obs_dim"]) if run_coordinated else None,
            "coordinated_local_obs_dim": int(coordinated_bundle["local_obs_dim"]) if run_coordinated else None,
            "local_only_obs_dim": int(local_bundle["local_obs_dim"]) if run_local else None,
            "baseline": baseline_metrics,
        }
        if run_local:
            report.update({
                "local_greedy": local_greedy_metrics,
                "local_stochastic_runs": local_stochastic_runs,
                "local_stochastic_mean": local_stochastic_mean,
                "comparison_local_greedy": _build_comparison(baseline_metrics, local_greedy_metrics),
                "comparison_local_stochastic_mean": _build_comparison(baseline_metrics, local_stochastic_mean),
            })
        if run_coordinated:
            report.update({
                "coordinated_greedy": coordinated_greedy_metrics,
                "coordinated_stochastic_runs": coordinated_stochastic_runs,
                "coordinated_stochastic_mean": coordinated_stochastic_mean,
                "comparison_coordinated_greedy": _build_comparison(baseline_metrics, coordinated_greedy_metrics),
                "comparison_coordinated_stochastic_mean": _build_comparison(baseline_metrics, coordinated_stochastic_mean),
            })
        report_path = os.path.join(report_dir, "comparison_coordinated_controller.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        _print_summary(report)
        print(f"Report saved to {report_path}")
    finally:
        try:
            if env.sumo.isLoaded():
                env.sumo.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()

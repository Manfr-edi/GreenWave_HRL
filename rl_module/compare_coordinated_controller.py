from __future__ import annotations

"""Evaluate RL controllers against a no-agent baseline.

The report can include the local-only controller, the coordinated controller, or
both. This lets the same route scenario answer two separate questions: whether
local control alone helps, and whether the global coordinator adds value.
"""

import argparse
import calendar
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import fields
from datetime import datetime, timedelta
from html import escape

import numpy as np
import torch
from tensordict import TensorDict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional runtime dependency
    matplotlib = None
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

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
    parse_taz_polygons,
    summarize_step_actions,
)
from rl_module.environment import CoordinatedTazTrafficEnv
from rl_module.policy import MultiDiscreteActorCritic, PriceAdapterActorCritic
from rl_module.settings import COORDINATOR, DEMAND, ENVIRONMENT, EXPERIMENT, LOCAL_CONTROLLER, PACKAGE_DIR
from rl_module.traffic_demand import (
    EpisodeDemand,
    _has_traffic_rows,
    _matches_day_filter,
    build_route_cache_folder,
    cleanup_sumo_output_files,
    ensure_routes,
    resolve_episode_demand,
)


DEFAULT_CHECKPOINT_PATH = os.path.join(PACKAGE_DIR, "checkpoints", "coordinated_controller", "coordinated_controller_best.pt")
DEFAULT_LOCAL_CHECKPOINT_PATH = os.path.join(PACKAGE_DIR, "checkpoints", "local_controller", "local_controller_best.pt")
DEFAULT_WEEKDAY_CHECKPOINT_PATH = os.path.join(
    PACKAGE_DIR,
    "saved_trainings",
    "weekdays_7_8",
    "checkpoints",
    "coordinated_controller",
    "coordinated_controller_best.pt",
)
DEFAULT_WEEKEND_CHECKPOINT_PATH = os.path.join(
    PACKAGE_DIR,
    "saved_trainings",
    "weekends_7_8",
    "checkpoints",
    "coordinated_controller",
    "coordinated_controller_best.pt",
)
DEFAULT_WEEKDAY_LOCAL_CHECKPOINT_PATH = os.path.join(
    PACKAGE_DIR,
    "saved_trainings",
    "weekdays_7_8",
    "checkpoints",
    "local_controller",
    "local_controller_best.pt",
)
DEFAULT_WEEKEND_LOCAL_CHECKPOINT_PATH = os.path.join(
    PACKAGE_DIR,
    "saved_trainings",
    "weekends_7_8",
    "checkpoints",
    "local_controller",
    "local_controller_best.pt",
)
DEFAULT_REPORT_ROOT = os.path.join(PACKAGE_DIR, "evaluation_reports", "coordinated_controller")
TRIPINFO_EMISSION_ATTRS = {
    "total_co2": "CO2_abs",
    "total_nox": "NOx_abs",
    "total_fuel": "fuel_abs",
    "total_co": "CO_abs",
}
EMISSION_OUTPUT_ATTRS = {
    "total_co2": "CO2",
    "total_nox": "NOx",
    "total_fuel": "fuel",
    "total_co": "CO",
}
COMPARISON_METRICS = (
    "avg_waiting_time",
    "total_waiting_time",
    "total_co2",
    "total_nox",
    "total_fuel",
    "total_co",
    "co2_mean_per_active_taz",
    "co2_max_taz_value",
    "co2_taz_range",
    "co2_taz_std",
    "co2_taz_cv",
)
SUPPORT_OVERVIEW_METRICS = (
    ("avg_waiting_time", "Attesa media"),
    ("total_waiting_time", "Attesa totale"),
    ("total_co2", "CO2 totale"),
    ("total_nox", "NOx totale"),
)
SUPPORT_CO2_BALANCE_METRICS = (
    ("mean_per_active_taz", "Media"),
    ("min_value", "Min"),
    ("max_value", "Max"),
    ("range", "Range"),
    ("std", "Std"),
    ("coefficient_of_variation", "CV"),
)
SUPPORT_PAIR_SPECS = (
    ("baseline_vs_local", "baseline", "local_greedy", "Baseline vs Locale"),
    ("baseline_vs_local_plus_coordinator", "baseline", "coordinated_greedy", "Baseline vs Locale + Coordinatore"),
    ("local_vs_local_plus_coordinator", "local_greedy", "coordinated_greedy", "Locale vs Locale + Coordinatore"),
)
SUPPORT_LABELS = {
    "baseline": "Baseline",
    "local_greedy": "Locale",
    "coordinated_greedy": "Locale + Coordinatore",
}
SUPPORT_COLORS = {
    "baseline": "#4c566a",
    "local_greedy": "#2a9d8f",
    "coordinated_greedy": "#e76f51",
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


def _empty_emissions_by_taz(taz_ids: list[str] | None) -> dict:
    return {
        str(taz_id): {**{metric_name: 0.0 for metric_name in EMISSION_OUTPUT_ATTRS}, "sample_count": 0}
        for taz_id in (taz_ids or [])
    }


def _xml_local_name(tag: str) -> str:
    return str(tag).rsplit("}", maxsplit=1)[-1]


def _prepare_taz_lookup(taz_ids: list[str] | None, taz_polygons: dict[str, list[tuple[float, float]]] | None) -> list[dict]:
    polygons = dict(taz_polygons or {})
    ordered_taz_ids = [str(taz_id) for taz_id in (taz_ids or polygons.keys())]
    lookup = []
    for taz_id in ordered_taz_ids:
        points = list(polygons.get(str(taz_id), []) or [])
        if len(points) < 3:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        lookup.append({
            "taz_id": str(taz_id),
            "points": points,
            "bbox": (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
        })
    return lookup


def _point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_intersection = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x <= x_intersection:
                inside = not inside
        j = i
    return inside


def _assign_taz_from_point(x: float, y: float, taz_lookup: list[dict]) -> str | None:
    for entry in taz_lookup:
        min_x, min_y, max_x, max_y = entry["bbox"]
        if x < min_x or x > max_x or y < min_y or y > max_y:
            continue
        if _point_in_polygon(x, y, entry["points"]):
            return str(entry["taz_id"])
    return None


def _parse_emission_output_by_taz(
    emission_output_path: str,
    taz_ids: list[str] | None,
    taz_polygons: dict[str, list[tuple[float, float]]] | None,
) -> dict:
    emissions_by_taz = _empty_emissions_by_taz(taz_ids)
    taz_lookup = _prepare_taz_lookup(taz_ids, taz_polygons)
    if not emission_output_path or not os.path.exists(emission_output_path) or not taz_lookup:
        return {
            "emissions_by_taz": emissions_by_taz,
            "assigned_sample_count": 0,
            "unassigned_sample_count": 0,
            "ok": False,
            "path": emission_output_path,
            "source": "emission_output_xy_polygon",
        }

    assigned_sample_count = 0
    unassigned_sample_count = 0
    try:
        for _, elem in ET.iterparse(emission_output_path, events=("end",)):
            tag = _xml_local_name(elem.tag)
            if tag == "vehicle":
                x_raw = elem.get("x")
                y_raw = elem.get("y")
                if x_raw is None or y_raw is None:
                    unassigned_sample_count += 1
                    elem.clear()
                    continue
                x = _safe_float(x_raw, float("nan"))
                y = _safe_float(y_raw, float("nan"))
                if not np.isfinite(x) or not np.isfinite(y):
                    unassigned_sample_count += 1
                    elem.clear()
                    continue
                taz_id = _assign_taz_from_point(x, y, taz_lookup)
                if taz_id is None:
                    unassigned_sample_count += 1
                    elem.clear()
                    continue
                bucket = emissions_by_taz.setdefault(
                    str(taz_id),
                    {**{metric_name: 0.0 for metric_name in EMISSION_OUTPUT_ATTRS}, "sample_count": 0},
                )
                for metric_name, attr_name in EMISSION_OUTPUT_ATTRS.items():
                    bucket[metric_name] = float(bucket.get(metric_name, 0.0)) + _safe_float(elem.get(attr_name), 0.0)
                bucket["sample_count"] = int(bucket.get("sample_count", 0)) + 1
                assigned_sample_count += 1
                elem.clear()
            elif tag == "timestep":
                elem.clear()
    except Exception:
        return {
            "emissions_by_taz": emissions_by_taz,
            "assigned_sample_count": int(assigned_sample_count),
            "unassigned_sample_count": int(unassigned_sample_count),
            "ok": False,
            "path": emission_output_path,
            "source": "emission_output_xy_polygon",
        }

    return {
        "emissions_by_taz": emissions_by_taz,
        "assigned_sample_count": int(assigned_sample_count),
        "unassigned_sample_count": int(unassigned_sample_count),
        "ok": bool(assigned_sample_count > 0),
        "path": emission_output_path,
        "source": "emission_output_xy_polygon",
    }


def _summarize_emissions_by_taz(emissions_by_taz: dict, taz_ids: list[str] | None) -> dict:
    ordered_taz_ids = [str(taz_id) for taz_id in (taz_ids or emissions_by_taz.keys())]
    summary = {}
    for metric_name in EMISSION_OUTPUT_ATTRS:
        active_values = [
            (taz_id, float((emissions_by_taz.get(taz_id, {}) or {}).get(metric_name, 0.0)))
            for taz_id in ordered_taz_ids
            if float((emissions_by_taz.get(taz_id, {}) or {}).get("sample_count", 0)) > 0.0
        ]
        if not active_values:
            summary[metric_name] = {
                "mean_per_active_taz": None,
                "min_taz": None,
                "min_value": None,
                "max_taz": None,
                "max_value": None,
                "range": None,
                "std": None,
                "coefficient_of_variation": None,
                "active_taz_count": 0,
                "configured_taz_count": int(len(ordered_taz_ids)),
            }
            continue
        values = np.array([value for _, value in active_values], dtype=np.float64)
        min_taz, min_value = min(active_values, key=lambda item: item[1])
        max_taz, max_value = max(active_values, key=lambda item: item[1])
        mean_value = float(values.mean())
        std_value = float(values.std())
        summary[metric_name] = {
            "mean_per_active_taz": mean_value,
            "min_taz": str(min_taz),
            "min_value": float(min_value),
            "max_taz": str(max_taz),
            "max_value": float(max_value),
            "range": float(max_value - min_value),
            "std": std_value,
            "coefficient_of_variation": float(std_value / mean_value) if abs(mean_value) > 1e-9 else None,
            "active_taz_count": int(len(active_values)),
            "configured_taz_count": int(len(ordered_taz_ids)),
        }
    return summary


def _co2_taz_aliases(emission_taz_summary: dict) -> dict:
    co2_summary = dict((emission_taz_summary or {}).get("total_co2", {}) or {})
    return {
        "co2_mean_per_active_taz": co2_summary.get("mean_per_active_taz"),
        "co2_min_taz": co2_summary.get("min_taz"),
        "co2_min_taz_value": co2_summary.get("min_value"),
        "co2_max_taz": co2_summary.get("max_taz"),
        "co2_max_taz_value": co2_summary.get("max_value"),
        "co2_taz_range": co2_summary.get("range"),
        "co2_taz_std": co2_summary.get("std"),
        "co2_taz_cv": co2_summary.get("coefficient_of_variation"),
        "emission_active_taz_count": int(co2_summary.get("active_taz_count", 0) or 0),
        "emission_configured_taz_count": int(co2_summary.get("configured_taz_count", 0) or 0),
    }


def _collect_run_metrics(
    run_root: str,
    waiting_components: dict | None = None,
    taz_ids: list[str] | None = None,
    taz_polygons: dict[str, list[tuple[float, float]]] | None = None,
) -> dict:
    output_dir = os.path.join(run_root, "output")
    waiting = dict(waiting_components or {})
    tripinfo_path = str(waiting.get("tripinfo_path") or os.path.join(output_dir, "tripinfos.xml"))
    emission_output_path = os.path.join(output_dir, "emission-output.xml")
    emissions = _parse_tripinfo_emission_totals(tripinfo_path)
    emissions_by_taz_payload = _parse_emission_output_by_taz(emission_output_path, taz_ids, taz_polygons)
    emission_taz_summary = _summarize_emissions_by_taz(emissions_by_taz_payload["emissions_by_taz"], taz_ids)
    return {
        "trip_count": int(waiting.get("trip_count", 0)),
        "tripinfo_count": int(emissions["tripinfo_count"]),
        "emission_trip_count": int(emissions["emission_trip_count"]),
        "avg_waiting_time": float(waiting.get("avg_waiting_time", 0.0)),
        "total_waiting_time": float(waiting.get("total_waiting_time", 0.0)),
        "waiting_metric_source": str(waiting.get("waiting_metric_source", "unknown")),
        "emission_metric_source": "tripinfo_emissions_abs",
        "emission_taz_metric_source": emissions_by_taz_payload["source"],
        "waiting_time_by_taz": dict(waiting.get("waiting_time_by_taz", {}) or {}),
        "vehicle_count_by_taz": dict(waiting.get("vehicle_count_by_taz", {}) or {}),
        "emissions_by_taz": emissions_by_taz_payload["emissions_by_taz"],
        "emission_taz_summary": emission_taz_summary,
        "emission_taz_parse_ok": bool(emissions_by_taz_payload["ok"]),
        "emission_assigned_sample_count": int(emissions_by_taz_payload["assigned_sample_count"]),
        "emission_unassigned_sample_count": int(emissions_by_taz_payload["unassigned_sample_count"]),
        **_co2_taz_aliases(emission_taz_summary),
        "total_co2": float(emissions["total_co2"]),
        "total_nox": float(emissions["total_nox"]),
        "total_fuel": float(emissions["total_fuel"]),
        "total_co": float(emissions["total_co"]),
        "ok": bool(waiting.get("parse_ok", False) and emissions["ok"]),
        "tripinfo_path": tripinfo_path,
        "emission_path": tripinfo_path,
        "emission_output_path": emission_output_path,
    }


def _cleanup_run_outputs(run_dir: str, keep_raw_output: bool):
    if keep_raw_output:
        return
    cleanup = cleanup_sumo_output_files(run_dir)
    if cleanup["errors"]:
        print(f"[WARN] SUMO output cleanup errors: {cleanup['errors'][:3]}")


def _evaluation_episode(simulation_date: str, hour: int, episode_index: int, demand_noise_override: float | None) -> EpisodeDemand:
    day = datetime.strptime(simulation_date, "%Y-%m-%d")
    episode = resolve_episode_demand(episode_index, day, hour, EXPERIMENT, DEMAND)
    if demand_noise_override is None:
        return episode
    vehicle_count = max(1, int(round(
        episode.base_vehicle_count * float(demand_noise_override) * episode.day_type_demand_factor
    )))
    route_folder_path = build_route_cache_folder(
        DEMAND.route_cache_root,
        episode.simulation_date,
        episode.timeslot_clean,
        episode.day_type,
        vehicle_count,
        float(demand_noise_override),
        episode.day_type_demand_factor,
        EXPERIMENT.global_seed,
        episode.random_trip_seed,
        episode.route_sampler_seed,
        DEMAND.route_sampler_threads,
    )
    return EpisodeDemand(
        episode_idx=int(episode_index),
        simulation_date=episode.simulation_date,
        day_type=episode.day_type,
        hour=episode.hour,
        timeslot=episode.timeslot,
        timeslot_clean=episode.timeslot_clean,
        hourly_profile_multiplier=episode.hourly_profile_multiplier,
        day_type_demand_factor=episode.day_type_demand_factor,
        base_vehicle_count=episode.base_vehicle_count,
        demand_noise=float(demand_noise_override),
        vehicle_count=vehicle_count,
        random_trip_seed=episode.random_trip_seed,
        route_sampler_seed=episode.route_sampler_seed,
        route_folder_path=route_folder_path,
        traffic_flow_file_path=episode.traffic_flow_file_path,
    )


def _global_context_tensor(episode: EpisodeDemand, reference_penalty: float | None, device: torch.device) -> torch.Tensor:
    reference_value = 0.0 if reference_penalty is None else float(np.clip(float(reference_penalty), -5.0, 5.0))
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
            price_max=max(price_bins),
        ).shape[-1]
    )
    coordinator_settings = dict(checkpoint.get("coordinator_settings", {}) or {})
    local_policy_class = str(checkpoint.get("local_policy_class", "MultiDiscreteActorCritic"))
    base_obs_dim = int(checkpoint.get("base_obs_dim", env.agent_obs_dim))
    price_context_dim = int(coordinator_settings.get("local_price_context_dim", COORDINATOR.local_price_context_dim))
    local_obs_dim = int(checkpoint.get("augmented_obs_dim", base_obs_dim + price_context_dim))
    if local_policy_class == "PriceAdapterActorCritic":
        local_policy = PriceAdapterActorCritic(
            base_obs_dim=base_obs_dim,
            price_context_dim=price_context_dim,
            act_dim=env.max_control_groups_per_taz,
            action_bins=local_action_bins,
            adapter_hidden_dim=int(coordinator_settings.get("price_adapter_hidden_dim", COORDINATOR.price_adapter_hidden_dim)),
            adapter_scale=float(coordinator_settings.get("price_adapter_scale", COORDINATOR.price_adapter_scale)),
        ).to(device)
    else:
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
        "local_policy_class": local_policy_class,
        "coordination_reference_mode": str(checkpoint.get("coordination_reference_mode", "no_agent_baseline")),
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


def _align_observation_width(observation: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad or truncate the last feature dimension to a target width."""

    target = int(target_dim)
    current = int(observation.shape[-1])
    if current == target:
        return observation
    if current > target:
        return observation[..., :target]
    return torch.nn.functional.pad(observation, (0, target - current))


def _build_price_context_tensor(
    local_observation: torch.Tensor,
    price_values: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    price_max: float | None = None,
) -> torch.Tensor:
    """Build the same 3-dim price context used by coordinated training."""

    prices = price_values.reshape(-1).to(device=local_observation.device, dtype=local_observation.dtype)
    adjacency = adjacency_matrix.to(device=local_observation.device, dtype=local_observation.dtype)
    if price_max is None:
        price_max = max(COORDINATION_PRICE_BINS)
    price_norm = prices / max(float(price_max), 1e-6)
    neighbor_price = adjacency.matmul(price_norm.unsqueeze(-1)).squeeze(-1) / adjacency.sum(dim=-1).clamp_min(1.0)
    return torch.stack([price_norm, neighbor_price, price_norm - neighbor_price], dim=-1)


def _prepare_coordinated_local_observation(
    local_observation: torch.Tensor,
    bundle: dict,
    price_values: torch.Tensor,
    adjacency_matrix: torch.Tensor,
) -> torch.Tensor:
    """Match local policy input width even when env and checkpoint schemas differ."""

    local_policy = bundle["local_policy"]
    local_policy_class = str(bundle.get("local_policy_class", ""))
    price_max = max(tuple(bundle.get("price_bins", COORDINATION_PRICE_BINS)))
    if local_policy_class == "PriceAdapterActorCritic":
        base_obs_dim = int(getattr(local_policy, "base_obs_dim", local_observation.shape[-1]))
        price_context_dim = int(getattr(local_policy, "price_context_dim", 3))
        base_obs = _align_observation_width(local_observation, base_obs_dim)
        price_context = _build_price_context_tensor(local_observation, price_values, adjacency_matrix, price_max=price_max)
        price_context = _align_observation_width(price_context, price_context_dim)
        return torch.cat([base_obs, price_context], dim=-1)

    augmented = augment_local_observation_with_prices(local_observation, price_values, adjacency_matrix, price_max=price_max)
    local_obs_dim = int(bundle.get("local_obs_dim", augmented.shape[-1]))
    return _align_observation_width(augmented, local_obs_dim)


def _accumulate_price_influence_pair(
    probe: dict,
    label: str,
    real_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    action_mask: torch.Tensor,
):
    valid_mask = action_mask.bool()
    valid_slots = int(valid_mask.sum().item())
    if valid_slots <= 0:
        return
    if label == "real_vs_zero":
        probe["valid_action_slots"] += valid_slots
    real_actions = real_logits.argmax(dim=-1)
    reference_actions = reference_logits.argmax(dim=-1)
    action_diff = int(((real_actions != reference_actions) & valid_mask).sum().item())
    probe[f"action_diff_{label}"] += action_diff
    expanded_mask = valid_mask.unsqueeze(-1).expand_as(real_logits)
    logit_delta = (real_logits - reference_logits).abs()[expanded_mask]
    if logit_delta.numel() <= 0:
        return
    probe[f"logit_delta_sum_{label}"] += float(logit_delta.sum().item())
    probe[f"logit_delta_count_{label}"] += int(logit_delta.numel())
    probe[f"max_abs_logit_delta_{label}"] = max(
        float(probe[f"max_abs_logit_delta_{label}"]),
        float(logit_delta.max().item()),
    )


def _finalize_price_influence_probe(probe: dict) -> dict:
    summary = {
        "steps": int(probe["steps"]),
        "nonzero_price_steps": int(probe["nonzero_price_steps"]),
        "valid_action_slots": int(probe["valid_action_slots"]),
        "unique_prices": sorted(float(value) for value in probe["unique_prices"]),
    }
    for label in ("real_vs_zero", "real_vs_base"):
        count = int(probe[f"logit_delta_count_{label}"])
        action_diff = int(probe[f"action_diff_{label}"])
        valid_slots = int(probe["valid_action_slots"])
        summary[f"action_diff_{label}"] = action_diff
        summary[f"action_diff_rate_{label}"] = float(action_diff / valid_slots) if valid_slots > 0 else None
        summary[f"mean_abs_logit_delta_{label}"] = (
            float(probe[f"logit_delta_sum_{label}"] / count) if count > 0 else None
        )
        summary[f"max_abs_logit_delta_{label}"] = float(probe[f"max_abs_logit_delta_{label}"])
    summary["price_channel_changed_actions"] = bool(summary["action_diff_real_vs_zero"] > 0)
    return summary


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
            local_obs = _align_observation_width(td["observation"], int(bundle.get("local_obs_dim", td["observation"].shape[-1])))
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


def _run_rl_episode(
    env,
    bundle: dict,
    adjacency_matrix: torch.Tensor,
    episode: EpisodeDemand,
    deterministic: bool,
    reference_penalty: float | None,
) -> dict:
    device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    num_taz = len(taz_ids)
    previous_flow_matrix = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=device)
    previous_prices = torch.zeros(num_taz, dtype=torch.float32, device=device)
    previous_action_summary = None
    step_prices = []
    step_diagnostics = []
    step_flows = []
    price_influence_probe = {
        "steps": 0,
        "nonzero_price_steps": 0,
        "valid_action_slots": 0,
        "unique_prices": set(),
        "action_diff_real_vs_zero": 0,
        "action_diff_real_vs_base": 0,
        "logit_delta_sum_real_vs_zero": 0.0,
        "logit_delta_sum_real_vs_base": 0.0,
        "logit_delta_count_real_vs_zero": 0,
        "logit_delta_count_real_vs_base": 0,
        "max_abs_logit_delta_real_vs_zero": 0.0,
        "max_abs_logit_delta_real_vs_base": 0.0,
    }
    if reference_penalty is not None:
        env.set_baseline_penalty(reference_penalty)
    else:
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
                extra_context=_global_context_tensor(episode, reference_penalty, device),
                config=bundle["observation_config"],
                price_max=max(tuple(bundle.get("price_bins", COORDINATION_PRICE_BINS))),
            )
            coordinator_action_mask = torch.ones((1, num_taz), dtype=torch.bool, device=device)
            coordinator_action_index, _, _ = bundle["coordinator"].act(coordinator_obs, coordinator_action_mask, deterministic=deterministic)
            price_values = bundle["coordinator"].action_values(coordinator_action_index).squeeze(0)
            policy_local_obs = _prepare_coordinated_local_observation(
                local_observation=local_obs,
                bundle=bundle,
                price_values=price_values,
                adjacency_matrix=adjacency_matrix,
            )
            local_action_index, _, _ = bundle["local_policy"].act(policy_local_obs, action_mask, deterministic=deterministic)
            price_influence_probe["steps"] += 1
            if bool((price_values.abs() > 1e-9).any().item()):
                price_influence_probe["nonzero_price_steps"] += 1
            for price in price_values.detach().cpu().reshape(-1).tolist():
                price_influence_probe["unique_prices"].add(round(float(price), 6))
            real_logits, _ = bundle["local_policy"].forward(policy_local_obs, action_mask)
            zero_policy_local_obs = _prepare_coordinated_local_observation(
                local_observation=local_obs,
                bundle=bundle,
                price_values=torch.zeros_like(price_values),
                adjacency_matrix=adjacency_matrix,
            )
            zero_logits, _ = bundle["local_policy"].forward(zero_policy_local_obs, action_mask)
            _accumulate_price_influence_pair(
                price_influence_probe,
                "real_vs_zero",
                real_logits,
                zero_logits,
                action_mask,
            )
            local_policy = bundle["local_policy"]
            if hasattr(local_policy, "base_policy"):
                base_obs_dim = int(getattr(local_policy, "base_obs_dim", local_obs.shape[-1]))
                base_obs = _align_observation_width(local_obs, base_obs_dim)
                base_logits, _ = local_policy.base_policy.forward(base_obs, action_mask)
                _accumulate_price_influence_pair(
                    price_influence_probe,
                    "real_vs_base",
                    real_logits,
                    base_logits,
                    action_mask,
                )
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
                baseline_penalty=reference_penalty if done else None,
                terminal_penalty=terminal_penalty if done else None,
                config=bundle["reward_config"],
                price_max=max(tuple(bundle.get("price_bins", COORDINATION_PRICE_BINS))),
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

    price_influence = _finalize_price_influence_probe(price_influence_probe)
    if price_influence["nonzero_price_steps"] > 0 and price_influence["action_diff_real_vs_zero"] == 0:
        print(
            "[WARN] Coordinator prices were non-zero but did not change any local greedy action "
            f"(max_abs_logit_delta={price_influence['max_abs_logit_delta_real_vs_zero']:.6g})."
        )

    return {
        "terminal_reward": dict((env.last_reward_components or {}).get("terminal_reward", {}) or {}),
        "group_action_details": dict((env.last_reward_components or {}).get("group_action_details", {}) or {}),
        "group_signal_by_taz": dict((env.last_reward_components or {}).get("group_signal_by_taz", {}) or {}),
        "coordination_rollout": {
            "enabled": True,
            "price_by_step": step_prices,
            "flow_by_step": step_flows,
            "step_diagnostics": step_diagnostics,
            "price_influence": price_influence,
            "episode_flow_by_taz": env.get_coordination_flow_diagnostics().get("episode_flow_by_taz", {}),
        },
    }


def _is_number(value) -> bool:
    try:
        parsed = float(value)
    except Exception:
        return False
    return bool(np.isfinite(parsed))


def _metric_comparison(reference_value: float, candidate_value: float, reference_name: str = "baseline", candidate_name: str = "rl") -> dict:
    reference = float(reference_value)
    candidate = float(candidate_value)
    delta = float(candidate - reference)
    improvement = float(reference - candidate)
    improvement_pct = float(improvement / reference * 100.0) if abs(reference) > 1e-9 else None
    return {
        "reference_name": str(reference_name),
        "candidate_name": str(candidate_name),
        "reference": reference,
        "candidate": candidate,
        "baseline": reference,
        "rl": candidate,
        "delta_candidate_minus_reference": delta,
        "delta_rl_minus_baseline": delta,
        "improvement_vs_reference": improvement,
        "improvement_vs_baseline": improvement,
        "improvement_pct": improvement_pct,
    }


def _build_comparison(baseline_metrics: dict, rl_metrics: dict, reference_name: str = "baseline", candidate_name: str = "rl") -> dict:
    return {
        metric: _metric_comparison(baseline_metrics[metric], rl_metrics[metric], reference_name, candidate_name)
        for metric in COMPARISON_METRICS
        if _is_number(baseline_metrics.get(metric)) and _is_number(rl_metrics.get(metric))
    }


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    taz_ids = sorted({
        str(taz_id)
        for metrics in metrics_list
        for taz_id in (metrics.get("emissions_by_taz", {}) or {}).keys()
    })

    def _mean(key: str) -> float:
        return float(np.mean([_safe_float(metrics.get(key), 0.0) for metrics in metrics_list]))

    def _mean_optional(key: str):
        values = [float(metrics.get(key)) for metrics in metrics_list if _is_number(metrics.get(key))]
        return float(np.mean(values)) if values else None

    emissions_by_taz = _empty_emissions_by_taz(taz_ids)
    for taz_id in taz_ids:
        buckets = [dict((metrics.get("emissions_by_taz", {}) or {}).get(taz_id, {}) or {}) for metrics in metrics_list]
        for metric_name in EMISSION_OUTPUT_ATTRS:
            emissions_by_taz[taz_id][metric_name] = float(np.mean([_safe_float(bucket.get(metric_name), 0.0) for bucket in buckets]))
        emissions_by_taz[taz_id]["sample_count"] = float(np.mean([_safe_float(bucket.get("sample_count"), 0.0) for bucket in buckets]))

    emission_taz_summary = _summarize_emissions_by_taz(emissions_by_taz, taz_ids)
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
        "emissions_by_taz": emissions_by_taz,
        "emission_taz_summary": emission_taz_summary,
        "emission_taz_parse_ok": bool(all(bool(metrics.get("emission_taz_parse_ok", False)) for metrics in metrics_list)),
        "emission_assigned_sample_count": _mean_optional("emission_assigned_sample_count"),
        "emission_unassigned_sample_count": _mean_optional("emission_unassigned_sample_count"),
        **_co2_taz_aliases(emission_taz_summary),
        "ok": bool(all(bool(metrics.get("ok", False)) for metrics in metrics_list)),
        "run_dirs": [str(metrics.get("run_dir", "")) for metrics in metrics_list],
    }


def _format_optional_float(value) -> str:
    return "n/a" if not _is_number(value) else f"{float(value):.4f}"


def _support_label(section_name: str) -> str:
    return str(SUPPORT_LABELS.get(section_name, section_name.replace("_", " ").title()))


def _support_color(section_name: str) -> str:
    return str(SUPPORT_COLORS.get(section_name, "#6c757d"))


def _support_plot_extension() -> str:
    return "png" if plt is not None else "svg"


def _pairwise_active_taz_ids(reference_metrics: dict, candidate_metrics: dict, metric_name: str = "total_co2") -> list[str]:
    taz_ids = []
    for metrics in (reference_metrics, candidate_metrics):
        emissions_by_taz = dict((metrics or {}).get("emissions_by_taz", {}) or {})
        for taz_id, values in emissions_by_taz.items():
            if float((values or {}).get("sample_count", 0.0)) <= 0.0:
                continue
            if str(taz_id) not in taz_ids:
                taz_ids.append(str(taz_id))
    taz_ids.sort(
        key=lambda taz_id: (
            -float(((reference_metrics.get("emissions_by_taz", {}) or {}).get(taz_id, {}) or {}).get(metric_name, 0.0)),
            str(taz_id),
        )
    )
    return taz_ids


def _co2_summary(metrics: dict) -> dict:
    return dict((metrics.get("emission_taz_summary", {}) or {}).get("total_co2", {}) or {})


def _svg_grouped_bar_panel(
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    category_labels: list[str],
    reference_values: list[float],
    candidate_values: list[float],
    reference_label: str,
    candidate_label: str,
    reference_color: str,
    candidate_color: str,
) -> str:
    outer_left = 70.0
    outer_right = 18.0
    outer_top = 40.0
    outer_bottom = 94.0
    plot_x = x + outer_left
    plot_y = y + outer_top
    plot_w = max(width - outer_left - outer_right, 10.0)
    plot_h = max(height - outer_top - outer_bottom, 10.0)
    values = [float(value) for value in reference_values + candidate_values if _is_number(value)]
    max_value = max(values) if values else 1.0
    max_value = max(max_value * 1.1, 1.0)
    tick_count = 4
    series_labels = [reference_label, candidate_label]
    series_colors = [reference_color, candidate_color]
    elements = [
        f'<text x="{x + width / 2.0:.1f}" y="{y + 22.0:.1f}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="18" fill="#1f2933">{escape(title)}</text>'
    ]

    for tick_idx in range(tick_count + 1):
        ratio = tick_idx / float(tick_count)
        yy = plot_y + plot_h - plot_h * ratio
        tick_value = max_value * ratio
        elements.append(
            f'<line x1="{plot_x:.1f}" y1="{yy:.1f}" x2="{plot_x + plot_w:.1f}" y2="{yy:.1f}" '
            f'stroke="#d9dee5" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{plot_x - 10.0:.1f}" y="{yy + 4.0:.1f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#52606d">{tick_value:.2f}</text>'
        )

    elements.append(
        f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_h:.1f}" '
        f'stroke="#7b8794" stroke-width="1.2"/>'
    )
    elements.append(
        f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}" '
        f'stroke="#7b8794" stroke-width="1.2"/>'
    )

    group_count = max(len(category_labels), 1)
    group_step = plot_w / float(group_count)
    bar_width = min(28.0, max(group_step * 0.28, 8.0))
    center_offset = bar_width * 0.62
    label_y = plot_y + plot_h + 54.0

    for index, category_label in enumerate(category_labels):
        center_x = plot_x + group_step * (index + 0.5)
        for series_idx, raw_value in enumerate((reference_values[index], candidate_values[index])):
            value = float(raw_value) if _is_number(raw_value) else 0.0
            bar_h = plot_h * max(value, 0.0) / max_value
            bar_center_x = center_x + (-center_offset if series_idx == 0 else center_offset)
            bar_x = bar_center_x - bar_width / 2.0
            bar_y = plot_y + plot_h - bar_h
            elements.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
                f'rx="2" ry="2" fill="{series_colors[series_idx]}"/>'
            )
        elements.append(
            f'<text x="{center_x:.1f}" y="{label_y:.1f}" text-anchor="end" '
            f'transform="rotate(-35 {center_x:.1f},{label_y:.1f})" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#1f2933">{escape(str(category_label))}</text>'
        )

    legend_x = plot_x + plot_w - 170.0
    legend_y = y + 16.0
    for index, (series_label, color) in enumerate(zip(series_labels, series_colors)):
        box_y = legend_y + index * 22.0
        elements.append(
            f'<rect x="{legend_x:.1f}" y="{box_y:.1f}" width="12" height="12" rx="2" ry="2" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{legend_x + 18.0:.1f}" y="{box_y + 10.0:.1f}" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#1f2933">{escape(series_label)}</text>'
        )
    return "\n".join(elements)


def _save_svg_document(path: str, width: float, height: float, body: str):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}">'
        f'<rect x="0" y="0" width="{int(width)}" height="{int(height)}" fill="#ffffff"/>'
        f"{body}</svg>"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def _plot_pairwise_overview(
    plot_path: str,
    pair_title: str,
    reference_section: str,
    reference_metrics: dict,
    candidate_section: str,
    candidate_metrics: dict,
):
    labels = [label for _, label in SUPPORT_OVERVIEW_METRICS]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.34
    reference_values = [float(reference_metrics[key]) for key, _ in SUPPORT_OVERVIEW_METRICS]
    candidate_values = [float(candidate_metrics[key]) for key, _ in SUPPORT_OVERVIEW_METRICS]

    reference_summary = _co2_summary(reference_metrics)
    candidate_summary = _co2_summary(candidate_metrics)
    balance_labels = [label for _, label in SUPPORT_CO2_BALANCE_METRICS]
    balance_x = np.arange(len(balance_labels), dtype=np.float64)
    reference_balance = [
        float(reference_summary.get(key)) if _is_number(reference_summary.get(key)) else np.nan
        for key, _ in SUPPORT_CO2_BALANCE_METRICS
    ]
    candidate_balance = [
        float(candidate_summary.get(key)) if _is_number(candidate_summary.get(key)) else np.nan
        for key, _ in SUPPORT_CO2_BALANCE_METRICS
    ]

    if plt is None:
        width = 1440.0
        height = 520.0
        body = "\n".join([
            _svg_grouped_bar_panel(
                24.0,
                16.0,
                680.0,
                480.0,
                pair_title,
                labels,
                reference_values,
                candidate_values,
                _support_label(reference_section),
                _support_label(candidate_section),
                _support_color(reference_section),
                _support_color(candidate_section),
            ),
            _svg_grouped_bar_panel(
                736.0,
                16.0,
                680.0,
                480.0,
                "CO2 per TAZ",
                balance_labels,
                reference_balance,
                candidate_balance,
                _support_label(reference_section),
                _support_label(candidate_section),
                _support_color(reference_section),
                _support_color(candidate_section),
            ),
        ])
        _save_svg_document(plot_path, width, height, body)
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(x - width / 2.0, reference_values, width=width, color=_support_color(reference_section), label=_support_label(reference_section))
    axes[0].bar(x + width / 2.0, candidate_values, width=width, color=_support_color(candidate_section), label=_support_label(candidate_section))
    axes[0].set_title(pair_title)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(
        balance_x - width / 2.0,
        reference_balance,
        width=width,
        color=_support_color(reference_section),
        label=_support_label(reference_section),
    )
    axes[1].bar(
        balance_x + width / 2.0,
        candidate_balance,
        width=width,
        color=_support_color(candidate_section),
        label=_support_label(candidate_section),
    )
    axes[1].set_title("CO2 per TAZ")
    axes[1].set_xticks(balance_x)
    axes[1].set_xticklabels(balance_labels, rotation=15, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pairwise_co2_by_taz(
    plot_path: str,
    pair_title: str,
    reference_section: str,
    reference_metrics: dict,
    candidate_section: str,
    candidate_metrics: dict,
):
    taz_ids = _pairwise_active_taz_ids(reference_metrics, candidate_metrics, metric_name="total_co2")
    if not taz_ids:
        return False

    reference_buckets = dict(reference_metrics.get("emissions_by_taz", {}) or {})
    candidate_buckets = dict(candidate_metrics.get("emissions_by_taz", {}) or {})
    reference_values = [float((reference_buckets.get(taz_id, {}) or {}).get("total_co2", 0.0)) for taz_id in taz_ids]
    candidate_values = [float((candidate_buckets.get(taz_id, {}) or {}).get("total_co2", 0.0)) for taz_id in taz_ids]

    if plt is None:
        width = max(1280.0, 90.0 * len(taz_ids))
        height = 520.0
        body = _svg_grouped_bar_panel(
            24.0,
            16.0,
            width - 48.0,
            height - 32.0,
            f"{pair_title} - CO2 per TAZ",
            taz_ids,
            reference_values,
            candidate_values,
            _support_label(reference_section),
            _support_label(candidate_section),
            _support_color(reference_section),
            _support_color(candidate_section),
        )
        _save_svg_document(plot_path, width, height, body)
        return True

    x = np.arange(len(taz_ids), dtype=np.float64)
    width = 0.42
    fig_width = max(11.0, 0.55 * len(taz_ids) + 4.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    ax.bar(x - width / 2.0, reference_values, width=width, color=_support_color(reference_section), label=_support_label(reference_section))
    ax.bar(x + width / 2.0, candidate_values, width=width, color=_support_color(candidate_section), label=_support_label(candidate_section))
    ax.set_title(f"{pair_title} - CO2 per TAZ")
    ax.set_xticks(x)
    ax.set_xticklabels(taz_ids, rotation=45, ha="right")
    ax.set_xlabel("TAZ")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_support_plots(report: dict, report_dir: str) -> list[dict]:
    plots_dir = os.path.join(report_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    if plt is None and MATPLOTLIB_IMPORT_ERROR is not None:
        print(f"[WARN] matplotlib not available, saving support plots as SVG: {MATPLOTLIB_IMPORT_ERROR}")
    generated = []
    extension = _support_plot_extension()
    for pair_key, reference_section, candidate_section, pair_title in SUPPORT_PAIR_SPECS:
        reference_metrics = report.get(reference_section)
        candidate_metrics = report.get(candidate_section)
        if not reference_metrics or not candidate_metrics:
            continue

        overview_path = os.path.join(plots_dir, f"{pair_key}_overview.{extension}")
        _plot_pairwise_overview(
            overview_path,
            pair_title,
            reference_section,
            reference_metrics,
            candidate_section,
            candidate_metrics,
        )
        generated.append({
            "plot_key": f"{pair_key}_overview",
            "plot_type": "overview",
            "pair_key": pair_key,
            "pair_title": pair_title,
            "models": [_support_label(reference_section), _support_label(candidate_section)],
                "path": overview_path,
                "relative_path": os.path.relpath(overview_path, report_dir),
            })

        taz_plot_path = os.path.join(plots_dir, f"{pair_key}_co2_by_taz.{extension}")
        if _plot_pairwise_co2_by_taz(
            taz_plot_path,
            pair_title,
            reference_section,
            reference_metrics,
            candidate_section,
            candidate_metrics,
        ):
            generated.append({
                "plot_key": f"{pair_key}_co2_by_taz",
                "plot_type": "co2_by_taz",
                "pair_key": pair_key,
                "pair_title": pair_title,
                "models": [_support_label(reference_section), _support_label(candidate_section)],
                "path": taz_plot_path,
                "relative_path": os.path.relpath(taz_plot_path, report_dir),
            })
    return generated


def _print_taz_emission_summary(report: dict):
    section_names = [
        "baseline",
        "local_greedy",
        "local_stochastic_mean",
        "coordinated_greedy",
        "coordinated_stochastic_mean",
    ]
    print("CO2 per TAZ")
    for section_name in section_names:
        metrics = report.get(section_name)
        if not metrics:
            continue
        summary = dict((metrics.get("emission_taz_summary", {}) or {}).get("total_co2", {}) or {})
        if not summary or not summary.get("active_taz_count"):
            print(f"{section_name}: n/a")
            continue
        print(
            f"{section_name}: mean={_format_optional_float(summary.get('mean_per_active_taz'))} | "
            f"min={summary.get('min_taz')}:{_format_optional_float(summary.get('min_value'))} | "
            f"max={summary.get('max_taz')}:{_format_optional_float(summary.get('max_value'))} | "
            f"range={_format_optional_float(summary.get('range'))} | std={_format_optional_float(summary.get('std'))} | "
            f"cv={_format_optional_float(summary.get('coefficient_of_variation'))} | "
            f"active_taz={summary.get('active_taz_count')}/{summary.get('configured_taz_count')}"
        )
    print("")


def _print_summary(report: dict):
    _print_taz_emission_summary(report)
    section_names = [
        "comparison_local_greedy",
        "comparison_local_stochastic_mean",
        "comparison_coordinated_greedy",
        "comparison_coordinated_stochastic_mean",
        "comparison_coordinated_vs_local_greedy",
        "comparison_coordinated_vs_local_stochastic_mean",
    ]
    for section_name in section_names:
        if section_name not in report:
            continue
        print(section_name)
        for metric_name, values in report[section_name].items():
            pct = values.get("improvement_pct")
            pct_text = "n/a" if pct is None else f"{pct:.2f}%"
            reference_name = values.get("reference_name", "baseline")
            candidate_name = values.get("candidate_name", "rl")
            print(
                f"{metric_name}: {reference_name}={values['reference']:.4f} | {candidate_name}={values['candidate']:.4f} | "
                f"delta={values['delta_candidate_minus_reference']:.4f} | "
                f"improvement={values['improvement_vs_reference']:.4f} | improvement_pct={pct_text}"
            )
        print("")


def _parse_date_arg(value: str, arg_name: str) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{arg_name} must have format YYYY-MM-DD") from exc


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(month).strip())
    if match is None:
        raise ValueError("--month must have format YYYY-MM")
    year = int(match.group(1))
    month_index = int(match.group(2))
    if month_index < 1 or month_index > 12:
        raise ValueError("--month must contain a month between 01 and 12")
    last_day = calendar.monthrange(year, month_index)[1]
    return datetime(year, month_index, 1), datetime(year, month_index, last_day)


def _timeslot_from_hour(hour: int) -> str:
    return f"{int(hour):02d}:00-{(int(hour) + 1) % 24:02d}:00"


def _parse_timeslot_list(timeslot: str | None, timeslots: str | None, default_hours: tuple[int, ...] | None = None) -> list[tuple[int, str, str]]:
    raw_slots = []
    if timeslots:
        raw_slots.extend(slot.strip() for slot in str(timeslots).split(",") if slot.strip())
    elif timeslot:
        raw_slots.append(str(timeslot).strip())
    elif default_hours:
        raw_slots.extend(_timeslot_from_hour(hour) for hour in default_hours)
    else:
        raise ValueError("Provide --timeslot or --timeslots.")

    parsed = []
    seen = set()
    for raw_slot in raw_slots:
        hour, clean = _parse_timeslot(raw_slot)
        if clean in seen:
            continue
        seen.add(clean)
        parsed.append((hour, raw_slot, clean))
    if not parsed:
        raise ValueError("No valid timeslots were provided.")
    return parsed


def _is_batch_mode(args) -> bool:
    return bool(args.month or args.months or args.start_date or args.end_date or args.timeslots)


def _validation_range(args) -> tuple[datetime, datetime]:
    if args.month and (args.months or args.start_date or args.end_date or args.date):
        raise ValueError("Use either --month, --months, or --start-date/--end-date, not more than one of them.")
    if args.months and (args.start_date or args.end_date or args.date):
        raise ValueError("Use either --months or --start-date/--end-date, not both.")
    if args.month:
        return _month_bounds(args.month)
    if bool(args.start_date) != bool(args.end_date):
        raise ValueError("--start-date and --end-date must be provided together.")
    if args.start_date and args.end_date:
        start_date = _parse_date_arg(args.start_date, "--start-date")
        end_date = _parse_date_arg(args.end_date, "--end-date")
        if end_date < start_date:
            raise ValueError("--end-date must be greater than or equal to --start-date.")
        return start_date, end_date
    if args.date:
        date = _parse_date_arg(args.date, "--date")
        return date, date
    raise ValueError("Provide --date for a single scenario, or --month / --start-date and --end-date for validation.")


def _validation_ranges(args) -> list[tuple[datetime, datetime]]:
    if not args.months:
        return [_validation_range(args)]
    if args.month or args.start_date or args.end_date or args.date:
        raise ValueError("Use either --months, --month, or --start-date/--end-date, not more than one of them.")
    ranges = [
        _month_bounds(token.strip())
        for token in str(args.months).split(",")
        if token.strip()
    ]
    if not ranges:
        raise ValueError("--months must contain at least one YYYY-MM token.")
    return ranges


def _build_validation_episodes(args) -> tuple[list[EpisodeDemand], dict]:
    ranges = _validation_ranges(args)
    start_date = min(start for start, _ in ranges)
    end_date = max(end for _, end in ranges)
    default_hours = EXPERIMENT.focus_hours if _is_batch_mode(args) else None
    parsed_timeslots = _parse_timeslot_list(args.timeslot, args.timeslots, default_hours=default_hours)
    episodes = []
    skipped_dates = []
    episode_idx = int(args.episode_index)
    seen_dates = set()
    for range_start, range_end in ranges:
        day = range_start
        while day <= range_end:
            date_text = day.strftime("%Y-%m-%d")
            if date_text in seen_dates:
                day += timedelta(days=1)
                continue
            seen_dates.add(date_text)
            if not _matches_day_filter(day, args.day_filter):
                day += timedelta(days=1)
                continue
            if not _has_traffic_rows(day):
                skipped_dates.append({"date": date_text, "reason": "missing_traffic_rows"})
                day += timedelta(days=1)
                continue
            for hour, _, _ in parsed_timeslots:
                episodes.append(_evaluation_episode(date_text, hour, episode_idx, args.demand_noise))
                episode_idx += 1
                if args.max_scenarios is not None and len(episodes) >= int(args.max_scenarios):
                    break
            if args.max_scenarios is not None and len(episodes) >= int(args.max_scenarios):
                break
            day += timedelta(days=1)
        if args.max_scenarios is not None and len(episodes) >= int(args.max_scenarios):
            break

    if not episodes:
        raise ValueError(
            "Validation scenario set is empty. Check date range, day filter, timeslots, and traffic-flow CSV coverage."
        )
    metadata = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "day_filter": str(args.day_filter),
        "months": [
            f"{start.year:04d}-{start.month:02d}"
            for start, _ in ranges
        ],
        "timeslots": [slot for _, slot, _ in parsed_timeslots],
        "skipped_dates": skipped_dates,
        "max_scenarios": args.max_scenarios,
        "episode_index_start": int(args.episode_index),
        "traffic_flow_files_by_year": {
            str(year): path
            for year, path in sorted({
                int(str(episode.simulation_date).split("-", maxsplit=1)[0]): episode.traffic_flow_file_path
                for episode in episodes
            }.items())
        },
    }
    return episodes, metadata


def _validation_name(args, metadata: dict) -> str:
    if args.validation_name:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.validation_name).strip()).strip("_") or "validation"
    if args.months:
        month_tag = "_".join(token.strip() for token in str(args.months).split(",") if token.strip())
        return f"months_{month_tag}"
    if args.month:
        return f"month_{args.month}"
    if metadata["start_date"] == metadata["end_date"]:
        return f"date_{metadata['start_date']}"
    return f"range_{metadata['start_date']}_{metadata['end_date']}"


def _checkpoint_mode(args) -> str:
    day_specific_args = (
        args.weekday_checkpoint,
        args.weekend_checkpoint,
        args.weekday_local_checkpoint,
        args.weekend_local_checkpoint,
    )
    if args.checkpoint_mode == "by-day-type" or any(day_specific_args):
        return "by-day-type"
    return "single"


def _coordinated_checkpoint_paths(args, mode: str, required_day_types: list[str]) -> dict[str, str]:
    if mode == "single":
        return {day_type: args.checkpoint for day_type in required_day_types}
    paths = {
        "weekday": args.weekday_checkpoint or DEFAULT_WEEKDAY_CHECKPOINT_PATH,
        "weekend": args.weekend_checkpoint or DEFAULT_WEEKEND_CHECKPOINT_PATH,
    }
    return {day_type: paths[day_type] for day_type in required_day_types}


def _local_checkpoint_paths(args, mode: str, required_day_types: list[str]) -> dict[str, str]:
    if mode == "single":
        return {day_type: args.local_checkpoint for day_type in required_day_types}
    paths = {
        "weekday": args.weekday_local_checkpoint or DEFAULT_WEEKDAY_LOCAL_CHECKPOINT_PATH,
        "weekend": args.weekend_local_checkpoint or DEFAULT_WEEKEND_LOCAL_CHECKPOINT_PATH,
    }
    return {day_type: paths[day_type] for day_type in required_day_types}


def _validate_checkpoint_paths(paths_by_day_type: dict[str, str], label: str, strict: bool) -> bool:
    missing = [
        f"{day_type}: {path}"
        for day_type, path in paths_by_day_type.items()
        if not path or not os.path.exists(path)
    ]
    if not missing:
        return True
    message = f"{label} checkpoint missing for " + "; ".join(missing)
    if strict:
        raise FileNotFoundError(message)
    print(f"[WARN] {message}")
    return False


def _load_checkpoint_entries(paths_by_day_type: dict[str, str]) -> dict[str, dict]:
    cache = {}
    entries = {}
    for day_type, path in paths_by_day_type.items():
        abs_path = os.path.abspath(path)
        if abs_path not in cache:
            cache[abs_path] = torch.load(abs_path, map_location="cpu", weights_only=True)
        entries[day_type] = {
            "day_type": day_type,
            "path": abs_path,
            "checkpoint": cache[abs_path],
        }
    return entries


def _checkpoint_taz_ids(entries: dict[str, dict]) -> list[str]:
    ids = []
    for entry in entries.values():
        for taz_id in entry.get("checkpoint", {}).get("taz_ids", []) or []:
            value = str(taz_id)
            if value not in ids:
                ids.append(value)
    return ids


def _selected_taz_ids_from_entries(*entry_maps: dict[str, dict]) -> list[str] | None:
    all_ids = []
    for entries in entry_maps:
        for taz_id in _checkpoint_taz_ids(entries or {}):
            if taz_id not in all_ids:
                all_ids.append(taz_id)
    return all_ids if len(all_ids) == 1 else None


def _add_local_bundles(entries: dict[str, dict], env: CoordinatedTazTrafficEnv) -> dict[str, dict]:
    for entry in entries.values():
        entry["bundle"] = _build_local_policy_bundle(entry["checkpoint"], env)
    return entries


def _add_coordinated_bundles(
    entries: dict[str, dict],
    env: CoordinatedTazTrafficEnv,
    adjacency_matrix: torch.Tensor,
) -> dict[str, dict]:
    for entry in entries.values():
        entry["bundle"] = _build_policy_bundle(entry["checkpoint"], env, adjacency_matrix)
    return entries


def _run_single_evaluation(
    args,
    episode: EpisodeDemand,
    run_local: bool,
    run_coordinated: bool,
    local_checkpoint: dict | None,
    checkpoint: dict | None,
    local_checkpoint_path: str | None,
    coordinated_checkpoint_path: str | None,
    sumo,
    planner,
    env: CoordinatedTazTrafficEnv,
    taz_ids: list[str],
    taz_polygons: dict[str, list[tuple[float, float]]],
    adjacency_matrix: torch.Tensor,
    adjacency_metadata: dict,
    local_bundle: dict | None,
    coordinated_bundle: dict | None,
) -> dict:
    report_dir = os.path.join(
        os.path.abspath(args.report_root),
        episode.simulation_date,
        episode.timeslot_clean,
        f"demand_{episode.vehicle_count}",
    )
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

    route_folder_path, generated = ensure_routes(episode, planner, DEMAND)
    print(f"[ROUTES] {'generated' if generated else 'cached'} | {route_folder_path}")

    print("[BASELINE] Running no-agent baseline.")
    sumo.changeRouteFilePath(route_folder_path)
    sumo.changeTypePath(baseline_run_dir)
    _, baseline_components = env.run_reference_episode_no_agent(hour=episode.hour)
    baseline_metrics = _collect_run_metrics(
        baseline_run_dir,
        waiting_components=baseline_components,
        taz_ids=taz_ids,
        taz_polygons=taz_polygons,
    )
    baseline_metrics["run_dir"] = baseline_run_dir
    baseline_penalty = float(baseline_components.get("penalty", 0.0)) if baseline_components.get("parse_ok", False) else None
    _cleanup_run_outputs(baseline_run_dir, args.keep_raw_output)

    local_greedy_metrics = None
    local_stochastic_runs = []
    local_stochastic_mean = None
    local_reference_penalty = None
    local_reference_components = {}
    if run_local:
        print("[LOCAL GREEDY] Running deterministic local-only controller.")
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(local_greedy_run_dir)
        rollout = _run_local_episode(env, local_bundle, episode, deterministic=True)
        local_greedy_metrics = _collect_run_metrics(
            local_greedy_run_dir,
            waiting_components=rollout["terminal_reward"],
            taz_ids=taz_ids,
            taz_polygons=taz_polygons,
        )
        local_greedy_metrics["run_dir"] = local_greedy_run_dir
        local_greedy_metrics["local_rollout"] = rollout["local_rollout"]
        local_greedy_metrics["group_action_details"] = rollout["group_action_details"]
        local_greedy_metrics["group_signal_by_taz"] = rollout["group_signal_by_taz"]
        local_greedy_metrics["terminal_reward"] = rollout["terminal_reward"]
        local_reference_components = dict(rollout["terminal_reward"])
        if bool(local_reference_components.get("parse_ok", False)):
            local_reference_penalty = float(local_reference_components.get("penalty", 0.0))
        _cleanup_run_outputs(local_greedy_run_dir, args.keep_raw_output)

        for run_idx in range(args.stochastic_runs):
            run_dir = os.path.join(local_stochastic_root_dir, f"run_{run_idx + 1:02d}")
            os.makedirs(os.path.join(run_dir, "output"), exist_ok=True)
            print(f"[LOCAL STOCHASTIC] Run {run_idx + 1}/{args.stochastic_runs}.")
            sumo.changeRouteFilePath(route_folder_path)
            sumo.changeTypePath(run_dir)
            rollout = _run_local_episode(env, local_bundle, episode, deterministic=False)
            metrics = _collect_run_metrics(
                run_dir,
                waiting_components=rollout["terminal_reward"],
                taz_ids=taz_ids,
                taz_polygons=taz_polygons,
            )
            metrics["run_dir"] = run_dir
            metrics["run_index"] = int(run_idx + 1)
            metrics["local_rollout"] = rollout["local_rollout"]
            local_stochastic_runs.append(metrics)
            _cleanup_run_outputs(run_dir, args.keep_raw_output)
        local_stochastic_mean = _aggregate_metrics(local_stochastic_runs)

    coordinated_greedy_metrics = None
    coordinated_stochastic_runs = []
    coordinated_stochastic_mean = None
    coordination_reference_mode = None
    coordination_reference_penalty = None
    if run_coordinated:
        coordination_reference_mode = str(coordinated_bundle.get("coordination_reference_mode", "no_agent_baseline"))
        if coordination_reference_mode in ("local_only", "local_reference") and local_reference_penalty is not None:
            coordination_reference_penalty = local_reference_penalty
        else:
            coordination_reference_mode = "no_agent_baseline"
            coordination_reference_penalty = baseline_penalty
        print("[COORDINATED GREEDY] Running deterministic coordinated controller.")
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(coordinated_greedy_run_dir)
        rollout = _run_rl_episode(
            env,
            coordinated_bundle,
            adjacency_matrix,
            episode,
            deterministic=True,
            reference_penalty=coordination_reference_penalty,
        )
        coordinated_greedy_metrics = _collect_run_metrics(
            coordinated_greedy_run_dir,
            waiting_components=rollout["terminal_reward"],
            taz_ids=taz_ids,
            taz_polygons=taz_polygons,
        )
        coordinated_greedy_metrics["run_dir"] = coordinated_greedy_run_dir
        coordinated_greedy_metrics["coordination_rollout"] = rollout["coordination_rollout"]
        coordinated_greedy_metrics["group_action_details"] = rollout["group_action_details"]
        coordinated_greedy_metrics["group_signal_by_taz"] = rollout["group_signal_by_taz"]
        coordinated_greedy_metrics["terminal_reward"] = rollout["terminal_reward"]
        _cleanup_run_outputs(coordinated_greedy_run_dir, args.keep_raw_output)

        for run_idx in range(args.stochastic_runs):
            run_dir = os.path.join(coordinated_stochastic_root_dir, f"run_{run_idx + 1:02d}")
            os.makedirs(os.path.join(run_dir, "output"), exist_ok=True)
            print(f"[COORDINATED STOCHASTIC] Run {run_idx + 1}/{args.stochastic_runs}.")
            sumo.changeRouteFilePath(route_folder_path)
            sumo.changeTypePath(run_dir)
            rollout = _run_rl_episode(
                env,
                coordinated_bundle,
                adjacency_matrix,
                episode,
                deterministic=False,
                reference_penalty=coordination_reference_penalty,
            )
            metrics = _collect_run_metrics(
                run_dir,
                waiting_components=rollout["terminal_reward"],
                taz_ids=taz_ids,
                taz_polygons=taz_polygons,
            )
            metrics["run_dir"] = run_dir
            metrics["run_index"] = int(run_idx + 1)
            metrics["coordination_rollout"] = rollout["coordination_rollout"]
            coordinated_stochastic_runs.append(metrics)
            _cleanup_run_outputs(run_dir, args.keep_raw_output)
        coordinated_stochastic_mean = _aggregate_metrics(coordinated_stochastic_runs)

    report = {
        "date": episode.simulation_date,
        "timeslot": episode.timeslot,
        "episode_index": int(episode.episode_idx),
        "episode_demand": episode.__dict__,
        "checkpoint_mode": _checkpoint_mode(args),
        "local_checkpoint_path": local_checkpoint_path if run_local else None,
        "local_checkpoint_episode": int(local_checkpoint.get("episode", -1)) if run_local else None,
        "coordinated_checkpoint_path": coordinated_checkpoint_path if run_coordinated else None,
        "coordinated_checkpoint_episode": int(checkpoint.get("episode", -1)) if run_coordinated else None,
        "architecture_name": "baseline_vs_local_vs_coordinated",
        "traffic_flow_file_path": episode.traffic_flow_file_path,
        "route_folder_path": route_folder_path,
        "route_generated": bool(generated),
        "taz_ids": taz_ids,
        "tls_ids": env.get_tls_ids(),
        "coordination_price_bins": list(coordinated_bundle["price_bins"]) if run_coordinated else None,
        "coordination_reference_mode": coordination_reference_mode if run_coordinated else None,
        "coordination_reference_penalty": coordination_reference_penalty if run_coordinated else None,
        "local_reference_penalty": local_reference_penalty if run_local else None,
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
            "comparison_local_greedy": _build_comparison(baseline_metrics, local_greedy_metrics, "baseline", "local_greedy"),
            "comparison_local_stochastic_mean": _build_comparison(
                baseline_metrics,
                local_stochastic_mean,
                "baseline",
                "local_stochastic_mean",
            ),
        })
    if run_coordinated:
        report.update({
            "coordinated_greedy": coordinated_greedy_metrics,
            "coordinated_stochastic_runs": coordinated_stochastic_runs,
            "coordinated_stochastic_mean": coordinated_stochastic_mean,
            "comparison_coordinated_greedy": _build_comparison(
                baseline_metrics,
                coordinated_greedy_metrics,
                "baseline",
                "coordinated_greedy",
            ),
            "comparison_coordinated_stochastic_mean": _build_comparison(
                baseline_metrics,
                coordinated_stochastic_mean,
                "baseline",
                "coordinated_stochastic_mean",
            ),
        })
    if run_local and run_coordinated:
        report.update({
            "comparison_coordinated_vs_local_greedy": _build_comparison(
                local_greedy_metrics,
                coordinated_greedy_metrics,
                "local_greedy",
                "coordinated_greedy",
            ),
            "comparison_coordinated_vs_local_stochastic_mean": _build_comparison(
                local_stochastic_mean,
                coordinated_stochastic_mean,
                "local_stochastic_mean",
                "coordinated_stochastic_mean",
            ),
        })
    report["support_plots"] = _save_support_plots(report, report_dir)
    report_path = os.path.join(report_dir, "comparison_coordinated_controller.json")
    report["report_path"] = report_path
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _print_summary(report)
    if report["support_plots"]:
        print(f"Support plots saved to {os.path.join(report_dir, 'plots')}")
    print(f"Report saved to {report_path}")
    return report


def _stats(values: list[float]) -> dict:
    clean = np.array([float(value) for value in values if _is_number(value)], dtype=np.float64)
    if clean.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "ci95_half_width": None,
        }
    std = float(clean.std(ddof=1)) if clean.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(clean.size)) if clean.size > 1 else 0.0
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": std,
        "median": float(np.median(clean)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "ci95_half_width": ci95,
    }


def _aggregate_report_section(reports: list[dict], section_name: str) -> dict | None:
    metrics_list = [
        dict(report.get(section_name) or {})
        for report in reports
        if isinstance(report.get(section_name), dict)
    ]
    if not metrics_list:
        return None
    return _aggregate_metrics(metrics_list)


def _comparison_distribution(reports: list[dict], section_name: str) -> dict:
    metric_names = sorted({
        metric_name
        for report in reports
        for metric_name in (report.get(section_name, {}) or {}).keys()
    })
    result = {}
    value_keys = (
        "reference",
        "candidate",
        "delta_candidate_minus_reference",
        "improvement_vs_reference",
        "improvement_pct",
    )
    for metric_name in metric_names:
        metric_rows = [
            dict((report.get(section_name, {}) or {}).get(metric_name, {}) or {})
            for report in reports
            if isinstance((report.get(section_name, {}) or {}).get(metric_name), dict)
        ]
        if not metric_rows:
            continue
        result[metric_name] = {
            key: _stats([row.get(key) for row in metric_rows if _is_number(row.get(key))])
            for key in value_keys
        }
        result[metric_name]["reference_name"] = metric_rows[0].get("reference_name")
        result[metric_name]["candidate_name"] = metric_rows[0].get("candidate_name")
    return result


def _build_validation_summary(reports: list[dict], metadata: dict, args, output_dir: str) -> dict:
    sections = (
        "baseline",
        "local_greedy",
        "local_stochastic_mean",
        "coordinated_greedy",
        "coordinated_stochastic_mean",
    )
    local_paths_by_day_type = {
        str((report.get("episode_demand", {}) or {}).get("day_type")): report.get("local_checkpoint_path")
        for report in reports
        if report.get("local_checkpoint_path")
    }
    coordinated_paths_by_day_type = {
        str((report.get("episode_demand", {}) or {}).get("day_type")): report.get("coordinated_checkpoint_path")
        for report in reports
        if report.get("coordinated_checkpoint_path")
    }
    aggregate = {
        "architecture_name": "validation_aggregate_baseline_vs_local_vs_coordinated",
        "validation": {
            **metadata,
            "scenario_count": int(len(reports)),
            "stochastic_runs": int(args.stochastic_runs),
            "demand_noise_override": args.demand_noise,
            "checkpoint_mode": _checkpoint_mode(args),
        },
        "local_checkpoint_paths_by_day_type": local_paths_by_day_type,
        "coordinated_checkpoint_paths_by_day_type": coordinated_paths_by_day_type,
        "scenario_reports": [
            {
                "date": report.get("date"),
                "timeslot": report.get("timeslot"),
                "episode_index": report.get("episode_index"),
                "day_type": (report.get("episode_demand", {}) or {}).get("day_type"),
                "vehicle_count": (report.get("episode_demand", {}) or {}).get("vehicle_count"),
                "traffic_flow_file_path": report.get("traffic_flow_file_path"),
                "local_checkpoint_path": report.get("local_checkpoint_path"),
                "coordinated_checkpoint_path": report.get("coordinated_checkpoint_path"),
                "report_path": report.get("report_path"),
                "route_folder_path": report.get("route_folder_path"),
                "route_generated": report.get("route_generated"),
            }
            for report in reports
        ],
    }

    for section_name in sections:
        section = _aggregate_report_section(reports, section_name)
        if section is not None:
            aggregate[section_name] = section

    if "baseline" in aggregate and "local_greedy" in aggregate:
        aggregate["comparison_local_greedy"] = _build_comparison(
            aggregate["baseline"],
            aggregate["local_greedy"],
            "baseline",
            "local_greedy",
        )
    if "baseline" in aggregate and "local_stochastic_mean" in aggregate:
        aggregate["comparison_local_stochastic_mean"] = _build_comparison(
            aggregate["baseline"],
            aggregate["local_stochastic_mean"],
            "baseline",
            "local_stochastic_mean",
        )
    if "baseline" in aggregate and "coordinated_greedy" in aggregate:
        aggregate["comparison_coordinated_greedy"] = _build_comparison(
            aggregate["baseline"],
            aggregate["coordinated_greedy"],
            "baseline",
            "coordinated_greedy",
        )
    if "baseline" in aggregate and "coordinated_stochastic_mean" in aggregate:
        aggregate["comparison_coordinated_stochastic_mean"] = _build_comparison(
            aggregate["baseline"],
            aggregate["coordinated_stochastic_mean"],
            "baseline",
            "coordinated_stochastic_mean",
        )
    if "local_greedy" in aggregate and "coordinated_greedy" in aggregate:
        aggregate["comparison_coordinated_vs_local_greedy"] = _build_comparison(
            aggregate["local_greedy"],
            aggregate["coordinated_greedy"],
            "local_greedy",
            "coordinated_greedy",
        )
    if "local_stochastic_mean" in aggregate and "coordinated_stochastic_mean" in aggregate:
        aggregate["comparison_coordinated_vs_local_stochastic_mean"] = _build_comparison(
            aggregate["local_stochastic_mean"],
            aggregate["coordinated_stochastic_mean"],
            "local_stochastic_mean",
            "coordinated_stochastic_mean",
        )

    comparison_sections = [
        "comparison_local_greedy",
        "comparison_local_stochastic_mean",
        "comparison_coordinated_greedy",
        "comparison_coordinated_stochastic_mean",
        "comparison_coordinated_vs_local_greedy",
        "comparison_coordinated_vs_local_stochastic_mean",
    ]
    aggregate["comparison_distributions"] = {
        section_name: _comparison_distribution(reports, section_name)
        for section_name in comparison_sections
        if any(section_name in report for report in reports)
    }
    aggregate["support_plots"] = _save_support_plots(aggregate, output_dir)
    return aggregate


def _metric_value(report: dict, section: str, metric: str):
    values = report.get(section)
    if not isinstance(values, dict):
        return ""
    value = values.get(metric)
    return value if _is_number(value) else ""


def _comparison_value(report: dict, section: str, metric: str, key: str = "improvement_pct"):
    values = (report.get(section, {}) or {}).get(metric)
    if not isinstance(values, dict):
        return ""
    value = values.get(key)
    return value if _is_number(value) else ""


def _write_validation_csv(reports: list[dict], csv_path: str):
    columns = [
        "date",
        "timeslot",
        "episode_index",
        "day_type",
        "vehicle_count",
        "local_checkpoint_path",
        "coordinated_checkpoint_path",
        "traffic_flow_file_path",
        "baseline_avg_waiting_time",
        "baseline_total_waiting_time",
        "baseline_total_co2",
        "baseline_total_nox",
        "local_greedy_avg_waiting_time",
        "local_greedy_total_waiting_time",
        "local_greedy_total_co2",
        "local_greedy_total_nox",
        "coordinated_greedy_avg_waiting_time",
        "coordinated_greedy_total_waiting_time",
        "coordinated_greedy_total_co2",
        "coordinated_greedy_total_nox",
        "local_greedy_co2_improvement_pct",
        "coordinated_greedy_co2_improvement_pct",
        "coordinated_vs_local_greedy_co2_improvement_pct",
        "report_path",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            demand = dict(report.get("episode_demand", {}) or {})
            writer.writerow({
                "date": report.get("date", ""),
                "timeslot": report.get("timeslot", ""),
                "episode_index": report.get("episode_index", ""),
                "day_type": demand.get("day_type", ""),
                "vehicle_count": demand.get("vehicle_count", ""),
                "local_checkpoint_path": report.get("local_checkpoint_path", ""),
                "coordinated_checkpoint_path": report.get("coordinated_checkpoint_path", ""),
                "traffic_flow_file_path": report.get("traffic_flow_file_path", ""),
                "baseline_avg_waiting_time": _metric_value(report, "baseline", "avg_waiting_time"),
                "baseline_total_waiting_time": _metric_value(report, "baseline", "total_waiting_time"),
                "baseline_total_co2": _metric_value(report, "baseline", "total_co2"),
                "baseline_total_nox": _metric_value(report, "baseline", "total_nox"),
                "local_greedy_avg_waiting_time": _metric_value(report, "local_greedy", "avg_waiting_time"),
                "local_greedy_total_waiting_time": _metric_value(report, "local_greedy", "total_waiting_time"),
                "local_greedy_total_co2": _metric_value(report, "local_greedy", "total_co2"),
                "local_greedy_total_nox": _metric_value(report, "local_greedy", "total_nox"),
                "coordinated_greedy_avg_waiting_time": _metric_value(report, "coordinated_greedy", "avg_waiting_time"),
                "coordinated_greedy_total_waiting_time": _metric_value(report, "coordinated_greedy", "total_waiting_time"),
                "coordinated_greedy_total_co2": _metric_value(report, "coordinated_greedy", "total_co2"),
                "coordinated_greedy_total_nox": _metric_value(report, "coordinated_greedy", "total_nox"),
                "local_greedy_co2_improvement_pct": _comparison_value(report, "comparison_local_greedy", "total_co2"),
                "coordinated_greedy_co2_improvement_pct": _comparison_value(report, "comparison_coordinated_greedy", "total_co2"),
                "coordinated_vs_local_greedy_co2_improvement_pct": _comparison_value(
                    report,
                    "comparison_coordinated_vs_local_greedy",
                    "total_co2",
                ),
                "report_path": report.get("report_path", ""),
            })


def _print_validation_summary(summary: dict):
    validation = summary.get("validation", {})
    print(
        f"Validation aggregate | scenarios={validation.get('scenario_count')} | "
        f"range={validation.get('start_date')}..{validation.get('end_date')} | "
        f"timeslots={','.join(validation.get('timeslots', []))}"
    )
    for section_name, metric_name in (
        ("comparison_local_greedy", "total_co2"),
        ("comparison_coordinated_greedy", "total_co2"),
        ("comparison_coordinated_vs_local_greedy", "total_co2"),
    ):
        stats = (
            summary.get("comparison_distributions", {})
            .get(section_name, {})
            .get(metric_name, {})
            .get("improvement_pct", {})
        )
        if not stats or stats.get("mean") is None:
            continue
        print(
            f"{section_name}.{metric_name}.improvement_pct: "
            f"mean={stats['mean']:.2f}% | std={stats['std']:.2f} | "
            f"ci95=+/-{stats['ci95_half_width']:.2f} | n={stats['count']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare no-agent baseline against RL controllers.")
    parser.add_argument("--date", default=None, help="Single simulation date in YYYY-MM-DD format.")
    parser.add_argument("--timeslot", default=None, help="Single timeslot in HH:MM-HH:MM format.")
    parser.add_argument("--month", default=None, help="Evaluate all available dates in a month, format YYYY-MM.")
    parser.add_argument("--months", default=None, help="Comma-separated months to evaluate, format YYYY-MM,YYYY-MM.")
    parser.add_argument("--start-date", default=None, help="Validation range start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default=None, help="Validation range end date in YYYY-MM-DD format.")
    parser.add_argument("--timeslots", default=None, help="Comma-separated full-hour slots for validation ranges.")
    parser.add_argument("--day-filter", default="all", help="Validation day filter: all, weekdays, weekends, or explicit weekdays.")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Optional cap for validation smoke runs.")
    parser.add_argument("--validation-name", default=None, help="Optional folder name for the aggregate validation report.")
    parser.add_argument(
        "--checkpoint-mode",
        choices=("single", "by-day-type"),
        default="single",
        help="Use one checkpoint for every scenario or choose weekday/weekend checkpoints from the calendar day.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Path to coordinated_controller checkpoint.")
    parser.add_argument("--local-checkpoint", default=DEFAULT_LOCAL_CHECKPOINT_PATH, help="Path to local_controller checkpoint.")
    parser.add_argument("--weekday-checkpoint", default=None, help="Coordinated checkpoint used for Monday-Friday in by-day-type mode.")
    parser.add_argument("--weekend-checkpoint", default=None, help="Coordinated checkpoint used for Saturday-Sunday in by-day-type mode.")
    parser.add_argument("--weekday-local-checkpoint", default=None, help="Local checkpoint used for Monday-Friday in by-day-type mode.")
    parser.add_argument("--weekend-local-checkpoint", default=None, help="Local checkpoint used for Saturday-Sunday in by-day-type mode.")
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT, help="Root folder for evaluation outputs.")
    parser.add_argument("--episode-index", type=int, default=0, help="First episode index used to reproduce demand noise.")
    parser.add_argument("--demand-noise", type=float, default=None, help="Optional explicit demand multiplier.")
    parser.add_argument("--stochastic-runs", type=int, default=1, help="Sampled-policy runs to average.")
    parser.add_argument("--skip-local", action="store_true", help="Skip local-only controller evaluation.")
    parser.add_argument("--skip-coordinated", action="store_true", help="Skip coordinated-controller evaluation.")
    parser.add_argument("--keep-raw-output", action="store_true", help="Keep bulky SUMO XML output files in report run folders.")
    args = parser.parse_args()

    if args.stochastic_runs < 1:
        raise ValueError("--stochastic-runs must be >= 1")
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise ValueError("--max-scenarios must be >= 1")
    if not _is_batch_mode(args) and (not args.date or not args.timeslot):
        raise ValueError("Single-scenario mode requires --date and --timeslot.")

    episodes, validation_metadata = _build_validation_episodes(args)
    batch_mode = len(episodes) > 1 or _is_batch_mode(args)
    required_day_types = sorted({episode.day_type for episode in episodes})
    checkpoint_mode = _checkpoint_mode(args)
    strict_checkpoint_paths = checkpoint_mode == "by-day-type"
    local_checkpoint_entries = {}
    coordinated_checkpoint_entries = {}

    if not args.skip_local:
        local_paths = _local_checkpoint_paths(args, checkpoint_mode, required_day_types)
        if _validate_checkpoint_paths(local_paths, "Local", strict=strict_checkpoint_paths):
            local_checkpoint_entries = _load_checkpoint_entries(local_paths)
    if not args.skip_coordinated:
        coordinated_paths = _coordinated_checkpoint_paths(args, checkpoint_mode, required_day_types)
        if _validate_checkpoint_paths(coordinated_paths, "Coordinated", strict=strict_checkpoint_paths):
            coordinated_checkpoint_entries = _load_checkpoint_entries(coordinated_paths)

    run_local = bool(local_checkpoint_entries)
    run_coordinated = bool(coordinated_checkpoint_entries)
    if not run_local and not run_coordinated:
        raise FileNotFoundError("No usable RL checkpoint found. Provide local and/or coordinated checkpoints.")
    selected_taz_ids = _selected_taz_ids_from_entries(coordinated_checkpoint_entries, local_checkpoint_entries)

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
        taz_ids = env.get_taz_ids()
        taz_polygons = parse_taz_polygons(constants.TAZ_ADDITIONAL_FILE_PATH)
        adjacency_matrix, adjacency_metadata = build_taz_adjacency_matrix(
            taz_ids=taz_ids,
            taz_additional_path=constants.TAZ_ADDITIONAL_FILE_PATH,
            distance_threshold=COORDINATOR.taz_adjacency_distance_threshold,
            fallback_k=COORDINATOR.taz_adjacency_fallback_k,
            device=torch.device(env.device),
        )
        if run_local:
            local_checkpoint_entries = _add_local_bundles(local_checkpoint_entries, env)
        if run_coordinated:
            coordinated_checkpoint_entries = _add_coordinated_bundles(coordinated_checkpoint_entries, env, adjacency_matrix)

        reports = []
        total = len(episodes)
        for index, episode in enumerate(episodes, start=1):
            local_entry = local_checkpoint_entries.get(episode.day_type) if run_local else None
            coordinated_entry = coordinated_checkpoint_entries.get(episode.day_type) if run_coordinated else None
            print(
                f"\n[SCENARIO {index}/{total}] date={episode.simulation_date} "
                f"timeslot={episode.timeslot} day_type={episode.day_type} vehicles={episode.vehicle_count}"
            )
            reports.append(
                _run_single_evaluation(
                    args=args,
                    episode=episode,
                    run_local=local_entry is not None,
                    run_coordinated=coordinated_entry is not None,
                    local_checkpoint=local_entry["checkpoint"] if local_entry else None,
                    checkpoint=coordinated_entry["checkpoint"] if coordinated_entry else None,
                    local_checkpoint_path=local_entry["path"] if local_entry else None,
                    coordinated_checkpoint_path=coordinated_entry["path"] if coordinated_entry else None,
                    sumo=sumo,
                    planner=planner,
                    env=env,
                    taz_ids=taz_ids,
                    taz_polygons=taz_polygons,
                    adjacency_matrix=adjacency_matrix,
                    adjacency_metadata=adjacency_metadata,
                    local_bundle=local_entry["bundle"] if local_entry else None,
                    coordinated_bundle=coordinated_entry["bundle"] if coordinated_entry else None,
                )
            )

        if batch_mode:
            validation_dir = os.path.join(os.path.abspath(args.report_root), "validation", _validation_name(args, validation_metadata))
            os.makedirs(validation_dir, exist_ok=True)
            summary = _build_validation_summary(reports, validation_metadata, args, validation_dir)
            summary_path = os.path.join(validation_dir, "validation_summary.json")
            csv_path = os.path.join(validation_dir, "validation_summary.csv")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            _write_validation_csv(reports, csv_path)
            _print_validation_summary(summary)
            print(f"Validation summary saved to {summary_path}")
            print(f"Validation CSV saved to {csv_path}")
    finally:
        try:
            if env.sumo.isLoaded():
                env.sumo.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()

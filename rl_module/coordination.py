"""Global coordination state, adjacency, and reward shaping utilities."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import torch


TAZ_FEATURE_DIM = 12
COORDINATION_PRICE_BINS = (0.0, 0.5, 1.0, 1.5, 2.0)


def _price_normalizer(price_max: float | None = None) -> float:
    if price_max is None:
        price_max = max(COORDINATION_PRICE_BINS)
    return max(float(price_max), 1e-6)


@dataclass
class CoordinationObservationConfig:
    """Feature switches for the coordinator observation."""

    per_taz_feature_dim: int = TAZ_FEATURE_DIM
    include_neighbor_mean: bool = True
    include_flow_features: bool = True
    include_previous_prices: bool = True
    include_city_mean: bool = True
    include_city_std: bool = True
    include_city_max: bool = True
    flow_ref: float = 100.0


@dataclass
class CoordinationRewardConfig:
    """Weights used by the emission-first coordinator reward."""

    reward_semantics_version: int = 2
    emission_mean_penalty_weight: float = 1.50
    worst_taz_emission_weight: float = 0.75
    emission_mean_delta_weight: float = 0.35
    emission_total_delta_weight: float = 0.10
    worst_taz_worsening_weight: float = 0.75
    emission_imbalance_weight: float = 0.75
    emission_spillover_weight: float = 0.75
    price_emission_alignment_weight: float = 0.10
    # The paired terminal delta is the only direct signal that the coordinated
    # policy beat the frozen local controller on the same route. Give it enough
    # weight to remain visible beside the accumulated dense penalties.
    terminal_delta_weight: float = 5.00
    city_delta_weight: float = 1.00
    imbalance_weight: float = 0.25
    spillover_weight: float = 0.85
    local_externality_weight: float = 0.75
    local_global_share_weight: float = 0.10
    local_bonus_clip: float = 1.5
    global_reward_clip: float = 4.0
    price_effort_weight: float = 0.01
    action_effort_weight: float = 0.25
    flow_ref: float = 100.0


DEFAULT_OBSERVATION_CONFIG = CoordinationObservationConfig()
DEFAULT_REWARD_CONFIG = CoordinationRewardConfig()


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros_like(values[0])
    return values.std(dim=0, unbiased=False)


def _zscore(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-6:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def parse_taz_polygons(taz_additional_path: str) -> dict[str, list[tuple[float, float]]]:
    """Read TAZ polygons from SUMO additional XML."""

    root = ET.parse(taz_additional_path).getroot()
    polygons = {}
    for taz in root.findall(".//taz"):
        taz_id = taz.get("id")
        shape = taz.get("shape")
        if not taz_id or not shape:
            continue
        points = []
        for raw_point in shape.split():
            try:
                x_raw, y_raw = raw_point.split(",", maxsplit=1)
                points.append((float(x_raw), float(y_raw)))
            except Exception:
                continue
        if len(points) >= 3:
            polygons[str(taz_id)] = points
    return polygons


def _polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    signed_area *= 0.5
    if abs(signed_area) < 1e-9:
        return float(sum(x for x, _ in points) / len(points)), float(sum(y for _, y in points) / len(points))
    return float(cx / (6.0 * signed_area)), float(cy / (6.0 * signed_area))


def _point_segment_distance(point, start, end) -> float:
    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _segments_distance(a0, a1, b0, b1) -> float:
    return min(
        _point_segment_distance(a0, b0, b1),
        _point_segment_distance(a1, b0, b1),
        _point_segment_distance(b0, a0, a1),
        _point_segment_distance(b1, a0, a1),
    )


def _simplify_polygon(points: list[tuple[float, float]], max_points: int = 180) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    stride = max(int(math.ceil(len(points) / float(max_points))), 1)
    return points[::stride]


def _polygon_boundary_distance(poly_a: list[tuple[float, float]], poly_b: list[tuple[float, float]]) -> float:
    poly_a = _simplify_polygon(poly_a)
    poly_b = _simplify_polygon(poly_b)
    best = float("inf")
    for idx_a, a0 in enumerate(poly_a):
        a1 = poly_a[(idx_a + 1) % len(poly_a)]
        for idx_b, b0 in enumerate(poly_b):
            b1 = poly_b[(idx_b + 1) % len(poly_b)]
            best = min(best, _segments_distance(a0, a1, b0, b1))
            if best <= 1e-6:
                return 0.0
    return float(best)


def build_taz_adjacency_matrix(
    taz_ids: list[str],
    taz_additional_path: str,
    distance_threshold: float = 80.0,
    fallback_k: int = 3,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict]:
    """Build a symmetric neighbor matrix from SUMO TAZ polygons."""

    polygons = parse_taz_polygons(taz_additional_path)
    n_taz = len(taz_ids)
    matrix = torch.zeros((n_taz, n_taz), dtype=torch.float32, device=device)
    centroids = {taz_id: _polygon_centroid(polygons.get(taz_id, [])) for taz_id in taz_ids}
    distances = {taz_id: {} for taz_id in taz_ids}

    for i, taz_i in enumerate(taz_ids):
        poly_i = polygons.get(taz_i, [])
        for j, taz_j in enumerate(taz_ids):
            if i == j:
                continue
            poly_j = polygons.get(taz_j, [])
            if poly_i and poly_j:
                distance = _polygon_boundary_distance(poly_i, poly_j)
            else:
                xi, yi = centroids[taz_i]
                xj, yj = centroids[taz_j]
                distance = math.hypot(xi - xj, yi - yj)
            distances[taz_i][taz_j] = float(distance)
            if distance <= float(distance_threshold):
                matrix[i, j] = 1.0

    for i, taz_id in enumerate(taz_ids):
        if float(matrix[i].sum().item()) > 0.0:
            continue
        nearest = sorted(distances[taz_id].items(), key=lambda item: item[1])[: max(int(fallback_k), 1)]
        for neighbor_id, _ in nearest:
            matrix[i, taz_ids.index(neighbor_id)] = 1.0

    matrix = torch.maximum(matrix, matrix.T)
    metadata = {
        "source": taz_additional_path,
        "distance_threshold": float(distance_threshold),
        "fallback_k": int(fallback_k),
        "neighbors_by_taz": {
            taz_id: [taz_ids[j] for j in range(n_taz) if bool(matrix[i, j].item())]
            for i, taz_id in enumerate(taz_ids)
        },
        "centroids_by_taz": {
            taz_id: [float(centroids[taz_id][0]), float(centroids[taz_id][1])]
            for taz_id in taz_ids
        },
    }
    return matrix, metadata


def extract_taz_features(local_observation: torch.Tensor, env) -> torch.Tensor:
    """Extract the inherited per-TAZ feature block from local observations."""

    start_idx = int(env.max_tls_per_taz * env.per_tls_feature_dim)
    end_idx = start_idx + int(env.per_taz_feature_dim)
    return local_observation[:, start_idx:end_idx]


def summarize_step_actions(taz_ids: list[str], applied_actions: torch.Tensor, applied_deltas: torch.Tensor, action_mask: torch.Tensor) -> dict:
    """Aggregate local action intensity into one summary per TAZ."""

    actions = applied_actions.detach().cpu()
    deltas = applied_deltas.detach().cpu()
    mask = action_mask.detach().cpu().bool()
    summary = {}
    for taz_idx, taz_id in enumerate(taz_ids):
        valid_actions = actions[taz_idx][mask[taz_idx]]
        valid_deltas = deltas[taz_idx][mask[taz_idx]]
        if valid_actions.numel() == 0:
            summary[taz_id] = {
                "avg_action_selected": 0.0,
                "avg_applied_duration_delta": 0.0,
                "applied_duration_nonzero_ratio": 0.0,
                "aggressive_action_ratio": 0.0,
            }
            continue
        summary[taz_id] = {
            "avg_action_selected": float(valid_actions.float().mean().item()),
            "avg_applied_duration_delta": float(valid_deltas.float().mean().item()),
            "applied_duration_nonzero_ratio": float((valid_deltas.abs() > 1e-3).float().mean().item()),
            "aggressive_action_ratio": float((valid_actions.abs() >= 10.0).float().mean().item()),
        }
    return summary


def _action_summary_tensor(taz_ids: list[str], action_summary_by_taz: dict | None, device: torch.device) -> torch.Tensor:
    if not action_summary_by_taz:
        return torch.zeros((len(taz_ids), 3), dtype=torch.float32, device=device)
    rows = []
    for taz_id in taz_ids:
        summary = dict(action_summary_by_taz.get(taz_id, {}) or {})
        rows.append([
            float(summary.get("avg_applied_duration_delta", 0.0)) / 10.0,
            float(summary.get("applied_duration_nonzero_ratio", 0.0)),
            float(summary.get("aggressive_action_ratio", 0.0)),
        ])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def build_coordination_observation(
    taz_features: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    previous_flow_matrix: torch.Tensor | None = None,
    previous_prices: torch.Tensor | None = None,
    previous_action_summary: dict | None = None,
    taz_ids: list[str] | None = None,
    extra_context: torch.Tensor | None = None,
    config: CoordinationObservationConfig | None = None,
    price_max: float | None = None,
) -> torch.Tensor:
    """Build the single global observation consumed by the coordinator."""

    cfg = config or DEFAULT_OBSERVATION_CONFIG
    features = taz_features[:, : int(cfg.per_taz_feature_dim)].float()
    device = features.device
    adjacency = adjacency_matrix.to(device=device, dtype=features.dtype)
    chunks = [features.reshape(-1)]

    if cfg.include_neighbor_mean:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        chunks.append((adjacency.matmul(features) / degree).reshape(-1))
    if cfg.include_flow_features:
        flow = torch.zeros((features.shape[0], features.shape[0]), dtype=features.dtype, device=device)
        if previous_flow_matrix is not None:
            flow = previous_flow_matrix.to(device=device, dtype=features.dtype)
        flow_ref = max(float(cfg.flow_ref), 1e-6)
        chunks.append(torch.stack([
            flow.sum(dim=-1) / flow_ref,
            flow.sum(dim=0) / flow_ref,
            (flow * adjacency).sum(dim=-1) / flow_ref,
            (flow * adjacency).sum(dim=0) / flow_ref,
        ], dim=-1).reshape(-1))
    if cfg.include_previous_prices:
        prices = torch.zeros(features.shape[0], dtype=features.dtype, device=device)
        if previous_prices is not None:
            prices = previous_prices.reshape(-1).to(device=device, dtype=features.dtype)
        price_norm = prices / _price_normalizer(price_max)
        neighbor_price = adjacency.matmul(price_norm.unsqueeze(-1)).squeeze(-1) / adjacency.sum(dim=-1).clamp_min(1.0)
        chunks.append(torch.stack([price_norm, neighbor_price, price_norm - neighbor_price], dim=-1).reshape(-1))
    if taz_ids is not None:
        chunks.append(_action_summary_tensor(taz_ids, previous_action_summary, device=device).reshape(-1))
    if cfg.include_city_mean:
        chunks.append(features.mean(dim=0))
    if cfg.include_city_std:
        chunks.append(_safe_std(features))
    if cfg.include_city_max:
        chunks.append(features.max(dim=0).values)
    if extra_context is not None:
        chunks.append(extra_context.reshape(-1).to(device=device, dtype=features.dtype))
    return torch.cat(chunks, dim=0).unsqueeze(0)


def augment_local_observation_with_prices(
    local_observation: torch.Tensor,
    price_values: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    price_max: float | None = None,
) -> torch.Tensor:
    """Append own price, neighbor price, and price gap to each local observation."""

    prices = price_values.reshape(-1).to(device=local_observation.device, dtype=local_observation.dtype)
    adjacency = adjacency_matrix.to(device=local_observation.device, dtype=local_observation.dtype)
    price_norm = prices / _price_normalizer(price_max)
    neighbor_price = adjacency.matmul(price_norm.unsqueeze(-1)).squeeze(-1) / adjacency.sum(dim=-1).clamp_min(1.0)
    return torch.cat([local_observation, torch.stack([price_norm, neighbor_price, price_norm - neighbor_price], dim=-1)], dim=-1)


def _dict_values_by_taz(taz_ids: list[str], values: dict | None, key: str) -> torch.Tensor:
    if not values:
        return torch.zeros(len(taz_ids), dtype=torch.float32)
    return torch.tensor([
        float(values.get(taz_id, {}).get(key, 0.0)) if isinstance(values.get(taz_id, {}), dict) else 0.0
        for taz_id in taz_ids
    ], dtype=torch.float32)


def _component_values_by_taz(
    taz_ids: list[str],
    reward_components: dict,
    component_key: str,
    fallback_key: str,
) -> torch.Tensor:
    details = dict((reward_components or {}).get("penalty_details_by_taz", {}) or {})
    fallback = dict((reward_components or {}).get(fallback_key, {}) or {})
    return torch.tensor([
        float((details.get(taz_id, {}) or {}).get(component_key, fallback.get(taz_id, 0.0)))
        for taz_id in taz_ids
    ], dtype=torch.float32)


def _delta_component_values_by_taz(
    taz_ids: list[str],
    reward_components: dict,
    delta_key: str,
    fallback_key: str = "delta_penalty_by_taz",
) -> torch.Tensor:
    values = dict((reward_components or {}).get(delta_key, {}) or {})
    fallback = dict((reward_components or {}).get(fallback_key, {}) or {})
    return torch.tensor([
        float(values.get(taz_id, fallback.get(taz_id, 0.0)))
        for taz_id in taz_ids
    ], dtype=torch.float32)


def compute_coordination_step_rewards(
    taz_ids: list[str],
    price_values: torch.Tensor,
    reward_components: dict,
    flow_matrix: torch.Tensor,
    action_summary_by_taz: dict | None,
    baseline_penalty: float | None = None,
    terminal_penalty: float | None = None,
    config: CoordinationRewardConfig | None = None,
    price_max: float | None = None,
) -> tuple[torch.Tensor, float, dict]:
    """Compute emission-focused local correction and scalar coordinator reward."""

    cfg = config or DEFAULT_REWARD_CONFIG
    device = price_values.device
    dtype = price_values.dtype
    n_taz = len(taz_ids)
    emission_penalties = _component_values_by_taz(taz_ids, reward_components, "emission_penalty", "penalty_by_taz").to(
        device=device,
        dtype=dtype,
    )
    co2_penalties = _component_values_by_taz(taz_ids, reward_components, "co2_penalty", "penalty_by_taz").to(
        device=device,
        dtype=dtype,
    )
    nox_penalties = _component_values_by_taz(taz_ids, reward_components, "nox_penalty", "penalty_by_taz").to(
        device=device,
        dtype=dtype,
    )
    delta_emission_penalties = _delta_component_values_by_taz(taz_ids, reward_components, "delta_emission_penalty_by_taz").to(
        device=device,
        dtype=dtype,
    )
    delta_co2_penalties = _delta_component_values_by_taz(taz_ids, reward_components, "delta_co2_penalty_by_taz").to(
        device=device,
        dtype=dtype,
    )
    delta_nox_penalties = _delta_component_values_by_taz(taz_ids, reward_components, "delta_nox_penalty_by_taz").to(
        device=device,
        dtype=dtype,
    )
    prices = price_values.reshape(-1).to(device=device, dtype=dtype)
    flow = flow_matrix.to(device=device, dtype=dtype)
    if flow.shape != (n_taz, n_taz):
        flow = torch.zeros((n_taz, n_taz), dtype=dtype, device=device)

    flow_share = flow / flow.sum(dim=-1, keepdim=True).clamp_min(1.0)
    emission_worsening = torch.clamp(-delta_emission_penalties, min=0.0)
    emission_gain = torch.clamp(delta_emission_penalties, min=0.0)
    normalized_price = prices / _price_normalizer(price_max)
    action_effort = _dict_values_by_taz(taz_ids, action_summary_by_taz, "aggressive_action_ratio").to(device=device, dtype=dtype)
    flow_intensity = torch.clamp(flow.sum(dim=-1) / max(float(cfg.flow_ref), 1e-6), min=0.0, max=2.0)
    if int(cfg.reward_semantics_version) <= 1:
        # Reproduce diagnostics for existing checkpoints.
        downstream_cost = flow_share.matmul(normalized_price * emission_worsening)
    else:
        # Externality is a property of downstream traffic/emissions.
        # Multiplying it by the coordinator's own price let the policy reduce
        # its penalty simply by selecting zero, without changing emissions.
        downstream_cost = flow_share.matmul(emission_worsening)
    emission_externality = torch.clamp(
        downstream_cost * (1.0 + cfg.action_effort_weight * action_effort) * (1.0 + 0.25 * flow_intensity)
        - 0.25 * emission_gain,
        min=0.0,
    )

    total_emission_delta = float(delta_emission_penalties.sum().item())
    mean_emission_delta = float(delta_emission_penalties.mean().item()) if delta_emission_penalties.numel() else 0.0
    total_emission_penalty = float(emission_penalties.sum().item())
    mean_emission_penalty = float(emission_penalties.mean().item()) if emission_penalties.numel() else 0.0
    max_emission_penalty = float(emission_penalties.max().item()) if emission_penalties.numel() else 0.0
    worst_taz_worsening = float(emission_worsening.max().item()) if emission_worsening.numel() else 0.0
    emission_imbalance = float(emission_penalties.std(unbiased=False).item()) if emission_penalties.numel() > 1 else 0.0
    emission_cv = float(emission_imbalance / mean_emission_penalty) if abs(mean_emission_penalty) > 1e-9 else 0.0
    emission_spillover = float(emission_externality.mean().item()) if emission_externality.numel() > 0 else 0.0
    price_effort = float(normalized_price.abs().mean().item()) if normalized_price.numel() > 0 else 0.0
    price_emission_alignment = (
        float((_zscore(normalized_price) * _zscore(emission_worsening)).mean().item())
        if n_taz > 1
        else 0.0
    )
    city_delta = total_emission_delta
    if baseline_penalty is not None and terminal_penalty is not None:
        terminal_reference_delta = float(baseline_penalty - terminal_penalty)
        city_reward_basis = "step_emission_delta+reference_terminal_delta"
    else:
        terminal_reference_delta = 0.0
        city_reward_basis = "step_emission_delta"
    global_reward = (
        - cfg.emission_mean_penalty_weight * mean_emission_penalty
        - cfg.worst_taz_emission_weight * max_emission_penalty
        + cfg.emission_mean_delta_weight * mean_emission_delta
        + cfg.emission_total_delta_weight * total_emission_delta
        + cfg.terminal_delta_weight * terminal_reference_delta
        - cfg.worst_taz_worsening_weight * worst_taz_worsening
        - cfg.emission_imbalance_weight * emission_imbalance
        - cfg.emission_spillover_weight * emission_spillover
        - cfg.price_effort_weight * price_effort
        + cfg.price_emission_alignment_weight * price_emission_alignment
    )
    global_reward = max(-cfg.global_reward_clip, min(cfg.global_reward_clip, float(global_reward)))
    local_adjustment = -cfg.local_externality_weight * emission_externality - cfg.local_global_share_weight * emission_penalties
    local_adjustment = torch.clamp(local_adjustment, -cfg.local_bonus_clip, cfg.local_bonus_clip)
    diagnostics = {
        "city_delta": float(city_delta),
        "city_reward_basis": city_reward_basis,
        "city_term": float(terminal_reference_delta),
        "terminal_reference_delta": float(terminal_reference_delta),
        "mean_emission_delta": float(mean_emission_delta),
        "total_emission_delta": float(total_emission_delta),
        "mean_emission_penalty": float(mean_emission_penalty),
        "total_emission_penalty": float(total_emission_penalty),
        "max_emission_penalty": float(max_emission_penalty),
        "worst_taz_worsening": float(worst_taz_worsening),
        "emission_imbalance": float(emission_imbalance),
        "emission_cv": float(emission_cv),
        "emission_spillover": float(emission_spillover),
        "imbalance": float(emission_imbalance),
        "spillover": float(emission_spillover),
        "price_effort": float(price_effort),
        "price_emission_alignment": float(price_emission_alignment),
        "alignment": float(price_emission_alignment),
        "global_reward": float(global_reward),
        "price_by_taz": {taz_id: float(prices[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "emission_penalty_by_taz": {taz_id: float(emission_penalties[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "co2_penalty_by_taz": {taz_id: float(co2_penalties[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "nox_penalty_by_taz": {taz_id: float(nox_penalties[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "delta_emission_penalty_by_taz": {taz_id: float(delta_emission_penalties[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "delta_co2_penalty_by_taz": {taz_id: float(delta_co2_penalties[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "delta_nox_penalty_by_taz": {taz_id: float(delta_nox_penalties[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "worsening_by_taz": {taz_id: float(emission_worsening[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "local_gain_by_taz": {taz_id: float(emission_gain[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "externality_by_taz": {taz_id: float(emission_externality[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "local_adjustment_by_taz": {taz_id: float(local_adjustment[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "outflow_by_taz": {taz_id: float(flow[idx].sum().item()) for idx, taz_id in enumerate(taz_ids)},
        "inflow_by_taz": {taz_id: float(flow[:, idx].sum().item()) for idx, taz_id in enumerate(taz_ids)},
    }
    return local_adjustment, float(global_reward), diagnostics

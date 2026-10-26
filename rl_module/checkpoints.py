"""Checkpoint and history helpers for RL training scripts."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics

import torch


def _finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def checkpoint_score_from_reference(
    terminal_penalty,
    reference_penalty,
    metric_name: str = "delta_penalty_vs_reference",
) -> tuple[float | None, str | None, float | None]:
    """Return the score used to pick the best checkpoint against a reference.

    The preferred score is the penalty delta against a reference run for the
    same route/hour. This avoids selecting checkpoints only because the current
    episode had low demand. When no reference is available, the terminal penalty
    is kept as a fallback so training can still save a best checkpoint.
    """

    terminal = _finite_float(terminal_penalty)
    if terminal is None:
        return None, None, None
    reference = _finite_float(reference_penalty)
    if reference is None:
        return terminal, "terminal_penalty", None
    delta = terminal - reference
    return delta, str(metric_name), delta


def checkpoint_score_from_baseline(
    terminal_penalty,
    baseline_penalty,
) -> tuple[float | None, str | None, float | None]:
    """Compatibility wrapper for no-agent baseline checkpoint scoring."""

    return checkpoint_score_from_reference(
        terminal_penalty,
        baseline_penalty,
        metric_name="delta_penalty_vs_baseline",
    )


def checkpoint_best_metadata(checkpoint: dict) -> dict:
    """Recover best-checkpoint metadata from a saved training checkpoint.

    Episode checkpoints store the best metadata known before their own update,
    while best checkpoints store the selected rollout itself. The fallback to
    current checkpoint fields keeps resume robust for older checkpoints.
    """

    def finite_from(*keys: str) -> float | None:
        for key in keys:
            value = _finite_float(checkpoint.get(key))
            if value is not None:
                return value
        return None

    def int_from(*keys: str) -> int:
        for key in keys:
            try:
                return int(checkpoint.get(key))
            except (TypeError, ValueError):
                continue
        return -1

    best_score = finite_from("best_checkpoint_score", "checkpoint_selection_score", "checkpoint_score")
    best_metric = (
        checkpoint.get("best_checkpoint_metric")
        or checkpoint.get("checkpoint_selection_metric")
        or checkpoint.get("checkpoint_score_metric")
    )
    return {
        "best_episode": int_from("best_episode", "episode"),
        "best_checkpoint_score": best_score,
        "best_checkpoint_metric": best_metric if best_score is not None else None,
        "best_terminal_penalty": finite_from("best_terminal_penalty", "terminal_penalty"),
        "best_baseline_penalty": finite_from("best_baseline_penalty", "baseline_penalty"),
        "best_local_reference_penalty": finite_from("best_local_reference_penalty", "local_reference_penalty"),
        "best_delta_penalty_vs_baseline": finite_from(
            "best_delta_penalty_vs_baseline",
            "delta_penalty_vs_baseline",
        ),
        "best_delta_penalty_vs_local_reference": finite_from(
            "best_delta_penalty_vs_local_reference",
            "delta_penalty_vs_local_reference",
        ),
        "best_checkpoint_qualified": bool(
            checkpoint.get(
                "best_checkpoint_qualified",
                checkpoint.get("checkpoint_promotion_qualified", False),
            )
        ),
    }


def rolling_checkpoint_score(
    history: list[dict],
    *,
    score_key: str = "checkpoint_score",
    window_size: int = 10,
    min_count: int = 5,
    aggregation: str = "mean",
) -> tuple[float | None, str | None, int]:
    """Return a rolling score for checkpoint-promotion stability checks.

    Lower scores are better. Median aggregation is useful for route-varying
    coordinated rollouts because it cannot be dominated by a single outlier.
    """

    scores = []
    for row in history:
        score = _finite_float(row.get(score_key))
        if score is not None:
            scores.append(float(score))
    if not scores:
        return None, None, 0
    recent = scores[-max(int(window_size), 1):]
    if len(recent) < max(int(min_count), 1):
        return None, None, len(recent)
    if aggregation == "mean":
        value = float(sum(recent) / len(recent))
    elif aggregation == "median":
        value = float(statistics.median(recent))
    else:
        raise ValueError(f"Unsupported rolling checkpoint aggregation: {aggregation!r}")
    return value, f"{score_key}_rolling_{aggregation}_{len(recent)}", len(recent)


def load_checkpoint(path: str) -> dict:
    """Load a Torch checkpoint with a clear path error."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_local_controller_weights(policy, checkpoint_path: str, base_obs_dim: int) -> bool:
    """Load a pretrained local controller into a price-augmented local policy."""

    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_obs_dim = checkpoint.get("base_obs_dim", checkpoint.get("obs_dim"))
    if checkpoint_obs_dim is not None and int(checkpoint_obs_dim) != int(base_obs_dim):
        print(
            "[WARN] Skipping local pretraining checkpoint with incompatible observation width: "
            f"checkpoint={int(checkpoint_obs_dim)} current={int(base_obs_dim)}. "
            "Retrain the local controller with the emission-free observation before using it as pretraining."
        )
        return False
    source = checkpoint.get(
        "local_controller_state_dict",
        checkpoint.get("model_state_dict", checkpoint.get("local_model_state_dict", checkpoint)),
    )
    try:
        if hasattr(policy, "load_base_state_dict"):
            policy.load_base_state_dict(source)
            return True
    except RuntimeError as exc:
        print(f"[WARN] Could not load local pretraining checkpoint: {exc}")
        return False
    target = policy.state_dict()
    patched = {}
    for key, target_tensor in target.items():
        # The coordinated local policy has extra price-context inputs. Existing
        # local weights are copied and new input columns remain zero-initialized.
        source_tensor = source.get(key)
        if source_tensor is None:
            patched[key] = target_tensor
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            patched[key] = source_tensor
            continue
        next_tensor = target_tensor.clone()
        if key == "extractor.net.0.weight" and source_tensor.ndim == 1:
            next_tensor[:base_obs_dim] = source_tensor[:base_obs_dim]
        elif key == "extractor.net.0.bias" and source_tensor.ndim == 1:
            next_tensor[:base_obs_dim] = source_tensor[:base_obs_dim]
        elif key == "extractor.net.1.weight" and source_tensor.ndim == 2:
            next_tensor[:, :base_obs_dim] = source_tensor[:, :base_obs_dim]
            next_tensor[:, base_obs_dim:] = 0.0
        else:
            print(f"[WARN] Skipping incompatible local checkpoint tensor: {key}")
        patched[key] = next_tensor
    try:
        policy.load_state_dict(patched)
    except RuntimeError as exc:
        print(f"[WARN] Could not load local pretraining checkpoint: {exc}")
        return False
    return True


def resume_coordinated_checkpoint(
    checkpoint_path: str,
    enabled: bool,
    resume_episode: bool,
    start_episode_override: int | None,
    local_policy,
    coordinator_policy,
    local_optimizer,
    coordinator_optimizer,
    local_scheduler,
    coordinator_scheduler,
) -> tuple[int, dict]:
    """Restore coordinated-controller state when a checkpoint is configured."""

    if not enabled:
        return int(start_episode_override or 0), {}
    checkpoint = load_checkpoint(checkpoint_path)
    expected_adapter_mode = getattr(local_policy, "adapter_mode", None)
    if expected_adapter_mode is not None:
        coordinator_settings = dict(checkpoint.get("coordinator_settings", {}) or {})
        checkpoint_adapter_mode = str(
            coordinator_settings.get("price_adapter_mode", "legacy")
        )
        if checkpoint_adapter_mode != str(expected_adapter_mode):
            raise ValueError(
                "Cannot resume a coordinated checkpoint with a different price-adapter "
                f"architecture: checkpoint={checkpoint_adapter_mode!r}, "
                f"current={expected_adapter_mode!r}. Start a fresh coordinated run "
                "from the pretrained local checkpoint instead."
            )
    if checkpoint.get("local_controller_state_dict") is not None:
        local_policy.load_state_dict(checkpoint["local_controller_state_dict"])
    if checkpoint.get("coordinator_state_dict") is not None:
        coordinator_policy.load_state_dict(checkpoint["coordinator_state_dict"])
    _try_load_optimizer(local_optimizer, checkpoint.get("local_optimizer_state_dict"), "local")
    _try_load_optimizer(coordinator_optimizer, checkpoint.get("coordinator_optimizer_state_dict"), "coordinator")
    _try_load_scheduler(local_scheduler, checkpoint.get("local_scheduler_state_dict"), "local")
    _try_load_scheduler(coordinator_scheduler, checkpoint.get("coordinator_scheduler_state_dict"), "coordinator")
    start_episode = int(checkpoint.get("episode", -1)) + 1 if resume_episode else 0
    if start_episode_override is not None:
        start_episode = int(start_episode_override)
    return start_episode, checkpoint_best_metadata(checkpoint) if start_episode > 0 else {}


def _try_load_optimizer(optimizer, state_dict, label: str):
    if optimizer is None or state_dict is None:
        return
    try:
        optimizer.load_state_dict(state_dict)
    except Exception as exc:
        print(f"[WARN] Could not load {label} optimizer state: {exc}")


def _try_load_scheduler(scheduler, state_dict, label: str):
    if scheduler is None or state_dict is None:
        return
    try:
        scheduler.load_state_dict(state_dict)
    except Exception as exc:
        print(f"[WARN] Could not load {label} scheduler state: {exc}")


def write_history_header(csv_path: str, csv_cols: list[str]):
    """Create a fresh CSV history file."""

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=csv_cols).writeheader()


def append_history(row: dict, csv_path: str, json_path: str, csv_cols: list[str], history: list[dict]):
    """Append one row to CSV and JSON histories."""

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    history.append(row)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_cols)
        writer.writerow({key: row.get(key, "") for key in csv_cols})
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def best_penalty_from_history(history: list[dict]) -> tuple[float, int]:
    """Recover the best terminal penalty from previous rows."""

    best_penalty = float("inf")
    best_episode = -1
    for row in history:
        if not bool(row.get("terminal_parse_ok", False)):
            continue
        try:
            penalty = float(row.get("terminal_penalty", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(penalty) and penalty < best_penalty:
            best_penalty = penalty
            best_episode = int(row.get("episode", -1))
    return best_penalty, best_episode

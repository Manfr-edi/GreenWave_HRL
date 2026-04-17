from __future__ import annotations

"""Checkpoint and history helpers for RL training scripts."""

import csv
import json
import math
import os

import torch


def load_checkpoint(path: str) -> dict:
    """Load a Torch checkpoint with a clear path error."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_local_controller_weights(policy, checkpoint_path: str, base_obs_dim: int):
    """Load a pretrained local controller into a price-augmented local policy."""

    checkpoint = load_checkpoint(checkpoint_path)
    source = checkpoint.get(
        "local_controller_state_dict",
        checkpoint.get("model_state_dict", checkpoint.get("local_model_state_dict", checkpoint)),
    )
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
    policy.load_state_dict(patched)


def maybe_resume_coordinated_checkpoint(
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
) -> int:
    """Restore coordinated-controller state when a checkpoint is configured."""

    if not enabled:
        return int(start_episode_override or 0)
    checkpoint = load_checkpoint(checkpoint_path)
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
    return start_episode


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

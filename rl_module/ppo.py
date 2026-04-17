from __future__ import annotations

"""Generic PPO utilities used by all RL trainers."""

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn


def set_global_seed(seed: int):
    """Seed all local RNGs while keeping deterministic reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute generalized advantage estimates for a rollout."""

    t_max, n_agents = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(n_agents, dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros(n_agents, dtype=rewards.dtype, device=rewards.device)
    for t_idx in reversed(range(t_max)):
        # A done mask prevents bootstrapping across episode boundaries.
        mask = 1.0 - dones[t_idx]
        delta = rewards[t_idx] + gamma * next_value * mask - values[t_idx]
        gae = delta + gamma * lam * mask * gae
        advantages[t_idx] = gae
        next_value = values[t_idx]
    return advantages, advantages + values


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize advantages without introducing NaNs on tiny batches."""

    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


def entropy_coef_now(episode_idx: int, total_episodes: int, start_value: float, end_value: float, warmup_ratio: float) -> float:
    """Linearly decay entropy after the configured warmup fraction."""

    progress = episode_idx / max(total_episodes - 1, 1)
    if progress <= warmup_ratio:
        return float(start_value)
    decay_progress = (progress - warmup_ratio) / max(1.0 - warmup_ratio, 1e-8)
    return float(start_value + (end_value - start_value) * decay_progress)


def ppo_update(
    policy,
    optimizer,
    obs,
    action_mask,
    act,
    logp_old,
    adv,
    ret,
    clip_ratio=0.2,
    ppo_epochs=4,
    minibatch_size=256,
    entropy_coef=0.01,
    value_coef=0.5,
    target_kl=None,
    max_grad_norm=0.5,
):
    """Run PPO over a flattened rollout batch."""

    n_samples = obs.shape[0]
    idx = torch.arange(n_samples, device=obs.device)
    batch_size = min(int(minibatch_size), n_samples)
    planned_updates = int(ppo_epochs) * max(int(math.ceil(n_samples / batch_size)), 1)
    performed_updates = 0
    clip_fracs = []
    kls = []
    entropies = []
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_total_loss = 0.0
    early_stop = False
    epochs_performed = 0

    for epoch_idx in range(int(ppo_epochs)):
        # Shuffle on the target device to avoid CPU/GPU indexing transfers.
        perm = idx[torch.randperm(n_samples, device=obs.device)]
        epoch_kls = []
        for start in range(0, n_samples, batch_size):
            mb_idx = perm[start:start + batch_size]
            logp, values, entropy = policy.evaluate(obs[mb_idx], act[mb_idx], action_mask[mb_idx])
            ratio = torch.exp(torch.clamp(logp - logp_old[mb_idx], -20.0, 20.0))
            surr1 = ratio * adv[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, ret[mb_idx])
            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            performed_updates += 1

            approx_kl = (logp_old[mb_idx] - logp).mean().item()
            epoch_kls.append(float(approx_kl))
            kls.append(float(approx_kl))
            clip_fracs.append(float((torch.abs(ratio - 1.0) > clip_ratio).float().mean().item()))
            entropies.append(float(entropy.mean().item()))
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_total_loss = float(total_loss.item())

        epochs_performed += 1
        if target_kl is not None and epoch_idx >= 1 and float(np.mean(epoch_kls)) > float(target_kl) * 1.25:
            early_stop = True
            break

    return {
        "total_loss": float(last_total_loss),
        "policy_loss": float(last_policy_loss),
        "value_loss": float(last_value_loss),
        "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "early_stop": bool(early_stop),
        "epochs_performed": int(epochs_performed),
        "epochs_planned": int(ppo_epochs),
        "update_fraction": float(performed_updates / max(planned_updates, 1)),
    }


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """Move optimizer state tensors after loading a CPU checkpoint."""

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def clone_state_dict_to_cpu(state_dict: dict) -> dict:
    """Create a device-agnostic copy of a state dict."""

    cloned = {}
    for key, value in state_dict.items():
        cloned[key] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    return cloned

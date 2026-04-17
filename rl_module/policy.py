from __future__ import annotations

"""Policy networks shared by local and coordinator training.

Both policies are multi-discrete actor-critics: the local controller predicts
one action for each controllable TLS group, while the coordinator predicts one
price-bin action for each TAZ.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class FeatureExtractor(nn.Module):
    """Shared encoder used by local and coordinator policies."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MultiDiscreteActorCritic(nn.Module):
    """Actor-critic with one categorical action head per controlled slot."""

    def __init__(self, obs_dim: int, act_dim: int, action_bins: tuple[float, ...]):
        super().__init__()
        self.num_bins = int(len(action_bins))
        self.zero_action_index = int(action_bins.index(0.0))
        self.extractor = FeatureExtractor(obs_dim)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.logits_head = nn.Linear(32, int(act_dim) * self.num_bins)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.act_dim = int(act_dim)
        self.register_buffer("action_bins", torch.tensor(action_bins, dtype=torch.float32), persistent=False)
        nn.init.orthogonal_(self.logits_head.weight, gain=0.01)
        nn.init.constant_(self.logits_head.bias, 0.0)

    def _apply_action_mask(self, logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        # Invalid slots are forced to the zero-action bin instead of being sampled.
        if action_mask is None:
            return logits
        valid_mask = action_mask.bool().unsqueeze(-1)
        forced_logits = torch.full_like(logits, -1e9)
        forced_logits[..., self.zero_action_index] = 0.0
        return torch.where(valid_mask, logits, forced_logits)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        features = self.extractor(obs)
        logits = self.logits_head(self.actor(features)).reshape(-1, self.act_dim, self.num_bins)
        value = self.critic(features).squeeze(-1)
        return self._apply_action_mask(logits, action_mask), value

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False):
        # Log-probabilities are summed across valid action slots to match PPO's
        # one-sample-per-TAZ or one-sample-per-global-step rollout format.
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        action_index = logits.argmax(dim=-1) if deterministic else dist.sample()
        valid_mask = action_mask.bool()
        action_index = torch.where(valid_mask, action_index, torch.full_like(action_index, self.zero_action_index))
        logp = (dist.log_prob(action_index) * valid_mask.to(dtype=logits.dtype)).sum(dim=-1)
        return action_index, logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor):
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        valid_mask = action_mask.bool()
        safe_actions = torch.where(valid_mask, actions.long(), torch.full_like(actions.long(), self.zero_action_index))
        logp = (dist.log_prob(safe_actions) * valid_mask.to(dtype=logits.dtype)).sum(dim=-1)
        entropy = (dist.entropy() * valid_mask.to(dtype=logits.dtype)).sum(dim=-1)
        return logp, value, entropy

    def action_values(self, action_indices: torch.Tensor) -> torch.Tensor:
        """Map discrete action indices back to physical action values."""

        safe_indices = action_indices.long().clamp(0, self.num_bins - 1)
        return self.action_bins[safe_indices]

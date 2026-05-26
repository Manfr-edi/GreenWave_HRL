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


class PriceAdapterActorCritic(nn.Module):
    """Local policy with a small trainable price adapter.

    The base local policy consumes the original local observation width. The
    coordinator price context is mapped both to an additive feature residual
    and to a direct action-logit bias. The direct bias keeps the coordinator's
    price signal observable even when the pretrained base policy is frozen.
    """

    def __init__(
        self,
        base_obs_dim: int,
        price_context_dim: int,
        act_dim: int,
        action_bins: tuple[float, ...],
        adapter_hidden_dim: int = 64,
        adapter_scale: float = 0.25,
    ):
        super().__init__()
        self.base_obs_dim = int(base_obs_dim)
        self.price_context_dim = int(price_context_dim)
        self.local_obs_dim = self.base_obs_dim + self.price_context_dim
        self.act_dim = int(act_dim)
        self.num_bins = int(len(action_bins))
        self.zero_action_index = int(action_bins.index(0.0))
        self.adapter_scale = float(adapter_scale)
        self.base_policy = MultiDiscreteActorCritic(self.base_obs_dim, self.act_dim, action_bins)
        self.price_adapter = nn.Sequential(
            nn.LayerNorm(self.price_context_dim),
            nn.Linear(self.price_context_dim, int(adapter_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(adapter_hidden_dim), 128),
        )
        self.price_logit_adapter = nn.Sequential(
            nn.LayerNorm(self.price_context_dim),
            nn.Linear(self.price_context_dim, int(adapter_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(adapter_hidden_dim), self.act_dim * self.num_bins),
        )
        self.register_buffer("action_bins", torch.tensor(action_bins, dtype=torch.float32), persistent=False)
        self._reset_adapter()

    def _reset_adapter(self):
        final = self.price_adapter[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        logit_final = self.price_logit_adapter[-1]
        if isinstance(logit_final, nn.Linear):
            nn.init.xavier_uniform_(logit_final.weight, gain=0.05)
            nn.init.zeros_(logit_final.bias)

    def load_base_state_dict(self, state_dict: dict):
        """Load a local-only checkpoint into the frozen base policy."""

        if any(str(key).startswith("base_policy.") for key in state_dict):
            self.load_state_dict(state_dict)
            return
        self.base_policy.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        patched_state = dict(state_dict)
        current_state = super().state_dict()
        legacy_price_adapter = (
            any(str(key).startswith("base_policy.") for key in patched_state)
            and any(str(key).startswith("price_adapter.") for key in patched_state)
            and not any(str(key).startswith("price_logit_adapter.") for key in patched_state)
        )
        if legacy_price_adapter:
            for key, tensor in current_state.items():
                if str(key).startswith("price_logit_adapter.") and key not in patched_state:
                    patched_state[key] = tensor
                    if str(key).startswith("price_logit_adapter.3."):
                        patched_state[key] = torch.zeros_like(tensor)
        return super().load_state_dict(patched_state, strict=strict, assign=assign)

    def configure_trainable(self, train_base_policy: bool, train_price_adapter: bool):
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(bool(train_base_policy))
        for parameter in self.price_adapter.parameters():
            parameter.requires_grad_(bool(train_price_adapter))
        for parameter in self.price_logit_adapter.parameters():
            parameter.requires_grad_(bool(train_price_adapter))

    def split_observation(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_obs = obs[..., : self.base_obs_dim]
        price_obs = obs[..., self.base_obs_dim : self.base_obs_dim + self.price_context_dim]
        if price_obs.shape[-1] < self.price_context_dim:
            pad_width = self.price_context_dim - int(price_obs.shape[-1])
            price_obs = nn.functional.pad(price_obs, (0, pad_width))
        return base_obs, price_obs

    def _apply_action_mask(self, logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        return self.base_policy._apply_action_mask(logits, action_mask)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        base_obs, price_obs = self.split_observation(obs)
        features = self.base_policy.extractor(base_obs)
        if self.price_context_dim > 0:
            features = features + self.adapter_scale * self.price_adapter(price_obs)
        logits = self.base_policy.logits_head(self.base_policy.actor(features)).reshape(-1, self.act_dim, self.num_bins)
        if self.price_context_dim > 0:
            logit_bias = self.price_logit_adapter(price_obs).reshape(-1, self.act_dim, self.num_bins)
            logits = logits + self.adapter_scale * logit_bias
        value = self.base_policy.critic(features).squeeze(-1)
        return self._apply_action_mask(logits, action_mask), value

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False):
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
        safe_indices = action_indices.long().clamp(0, self.num_bins - 1)
        return self.action_bins[safe_indices]

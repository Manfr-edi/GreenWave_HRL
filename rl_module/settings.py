from __future__ import annotations

"""Central configuration for the RL package.

The goal is to keep experiment choices out of the training loops. Training
scripts import immutable settings from here, while output paths are derived from
the architecture name so local-only and coordinated runs never overwrite each
other.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime

from libraries.constants import SUMO_PATH


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TAZ_RL_DIR = os.path.dirname(PACKAGE_DIR)


@dataclass(frozen=True)
class ExperimentSettings:
    """Episode count, output names, and reproducibility settings for training."""

    name: str = "coordinated_controller"
    start_date: datetime = datetime(2024, 2, 1)
    # This is the target number of training episodes, not a calendar-day window.
    train_days: int = 200
    focus_hours: tuple[int, ...] = (7, 8)
    # Supported values: all, weekdays, weekends, or explicit weekdays like "mon-wed-sat".
    day_filter: str = "mon-wed-sat"
    train_on_same_day: bool = False
    shuffle_episode_schedule: bool = True
    global_seed: int = 42
    selected_taz_id: str | None = None

    @property
    def checkpoint_dir(self) -> str:
        # Every architecture gets an isolated checkpoint folder.
        return os.path.join(PACKAGE_DIR, "checkpoints", self.name)

    @property
    def detail_dir(self) -> str:
        # Episode-level JSON diagnostics are separated from compact histories.
        return os.path.join(PACKAGE_DIR, "details", self.name)

    @property
    def history_csv_path(self) -> str:
        return os.path.join(PACKAGE_DIR, "history", f"{self.name}.csv")

    @property
    def history_json_path(self) -> str:
        return os.path.join(PACKAGE_DIR, "history", f"{self.name}.json")


@dataclass(frozen=True)
class DemandSettings:
    """Demand profile and deterministic randomization rules."""

    base_demand: int = 5000
    hourly_profile: dict[int, float] = field(default_factory=lambda: {
        0: 0.2, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.2,
        5: 0.4, 6: 0.7, 7: 1.2, 8: 1.5,
        9: 1.0, 10: 0.8, 11: 0.9,
        12: 1.1, 13: 1.0, 14: 0.9, 15: 1.0,
        16: 1.3, 17: 1.6, 18: 1.4, 19: 1.0,
        20: 0.8, 21: 0.6, 22: 0.4, 23: 0.3,
    })
    # Randomness is reproducible: values are sampled from global_seed and episode metadata.
    demand_noise_range: tuple[float, float] = (0.85, 1.15)
    route_cache_root: str = os.path.join(SUMO_PATH, "routes_rl_module_randomized")
    reuse_cached_routes: bool = True
    random_trip_seed: int = 42
    route_sampler_seed: int = 42
    route_sampler_threads: int = 1
    # Keep False for now because the user asked not to change the seed; demand noise
    # and deterministic schedule shuffling already make scenarios less repetitive.
    vary_route_seeds_by_episode: bool = False


@dataclass(frozen=True)
class EnvironmentSettings:
    """SUMO environment reward and normalization constants."""

    speed_norm: float = 10.0
    jam_norm: float = 10.0
    warmupSteps: int = 0
    cooldownSteps: int = 12
    min_green: int = 25
    max_green: int = 120
    metric_clip: float = 2.0
    reward_clip: float = 4.0
    dense_abs_penalty_weight: float = 1.0
    dense_delta_reward_weight: float = 0.15
    terminal_bonus_weight: float = 1.0
    comparison_reward_enabled: bool = True
    waiting_reward_weight: float = 0.0
    emission_reward_weight: float = 1.0
    jam_reward_weight: float = 0.0
    emission_co2_mix_weight: float = 0.50
    emission_nox_mix_weight: float = 0.50
    tls_waiting_time_ref: float = 1000.0
    tls_co2_ref: float = 80000.0
    tls_nox_ref: float = 40.0
    tls_active_vehicle_ref: float = 80.0
    taz_waiting_time_ref: float = 6000.0
    taz_co2_ref: float = 300000.0
    taz_nox_ref: float = 150.0
    terminal_waiting_time_ref: float = 500000.0
    terminal_co2_ref: float = 5000000000.0
    terminal_nox_ref: float = 1800000.0
    tls_veh_total_ref: float = 120.0
    tls_pressure_ref: float = 10000.0
    taz_veh_total_ref: float = 2500.0
    taz_std_occupancy_ref: float = 0.25
    max_jam_len_ref: float = 120.0
    waiting_time_memory: int = 3600
    action_signal_threshold: float = 0.0
    sumo_thread_rngs: int = 1
    coordination_distance: float = 350.0
    coordination_path_length: float = 900.0
    coordination_group_max_size: int = 3
    group_axis_align_only: bool = True

    def to_env_kwargs(self, sumo_seed: int) -> dict:
        # The SUMO environment expects keyword arguments, not a dataclass object.
        data = dict(self.__dict__)
        data["sumo_seed"] = int(sumo_seed)
        return data


@dataclass(frozen=True)
class LocalControllerSettings:
    """PPO settings for the traffic-light controller."""

    action_bins: tuple[float, ...] = (-10.0, 0.0, 10.0)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2e-4
    clip_ratio: float = 0.12
    entropy_coef: float = 0.008
    entropy_coef_final: float = 0.001
    entropy_warmup_ratio: float = 0.25
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    ppo_epochs: int = 4
    minibatch_size: int = 64
    load_pretrained: bool = True
    pretrained_checkpoint_path: str = os.path.join(PACKAGE_DIR, "checkpoints", "local_controller", "local_controller_best.pt")
    train_local_policy: bool = True


@dataclass(frozen=True)
class CoordinatorSettings:
    """PPO and coordination settings for the global TAZ coordinator."""

    price_bins: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    local_price_context_dim: int = 3
    taz_adjacency_distance_threshold: float = 80.0
    taz_adjacency_fallback_k: int = 3
    learning_rate: float = 1e-4
    clip_ratio: float = 0.10
    entropy_coef: float = 0.010
    entropy_coef_final: float = 0.002
    value_coef: float = 0.5
    target_kl: float = 0.015
    ppo_epochs: int = 4
    minibatch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95


EXPERIMENT = ExperimentSettings()
DEMAND = DemandSettings()
ENVIRONMENT = EnvironmentSettings()
LOCAL_CONTROLLER = LocalControllerSettings()
COORDINATOR = CoordinatorSettings()

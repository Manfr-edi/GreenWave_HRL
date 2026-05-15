from __future__ import annotations

"""SUMO TAZ traffic environments used by the RL package.

The runtime environment is intentionally self-contained in this module and uses
role-based class names:

- ``LocalTazTrafficEnv`` builds the per-TAZ observation, reward, and one-action-per-TLS control surface.
- ``GroupedTazTrafficEnv`` replaces raw TLS actions with topology-aware TLS groups inside each TAZ.
- ``CoordinatedTazTrafficEnv`` adds online TAZ-to-TAZ flow tracking for the global coordinator.
"""

import heapq
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Optional

import libtraci
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

try:
    from torchrl.data import Bounded as BoundedSpec, Unbounded as UnboundedSpec
except Exception:
    from torchrl.data import BoundedTensorSpec as BoundedSpec
    from torchrl.data import UnboundedContinuousTensorSpec as UnboundedSpec

from libraries.constants import SUMO_NET_FILE_PATH, SUMO_NETWORK_PATH, TAZ_FILE

def aggregate_tls_metrics(detector_records):
    """Collapse raw E2 detector rows for one TLS into a compact metric dict."""

    if not detector_records:
        return {
            "veh_total": 0.0,
            "mean_speed": 0.0,
            "mean_occupancy": 0.0,
            "max_jam": 0.0,
            "pressure": 0.0
        }

    veh_total = sum(float(r.get("nVeh", 0.0)) for r in detector_records)

    if veh_total > 0:
        mean_speed = (
            sum(float(r.get("meanSpeed", 0.0)) * float(r.get("nVeh", 0.0)) for r in detector_records)
            / veh_total
        )
    else:
        mean_speed = 0.0

    mean_occupancy = sum(float(r.get("occupancy", 0.0)) for r in detector_records) / len(detector_records)
    max_jam = max(float(r.get("maxJamLength", 0.0)) for r in detector_records)

    # Pressure is a lightweight congestion proxy: many vehicles with low speed
    # should be treated as a stronger signal than the same count at high speed.
    pressure = veh_total * (1 - mean_speed)

    return {
        "veh_total": veh_total,
        "mean_speed": mean_speed,
        "mean_occupancy": mean_occupancy,
        "max_jam": max_jam,
        "pressure": pressure
    }


def aggregate_taz_metrics(tls_metrics):
    """Aggregate the already-normalized TLS metrics into one TAZ-level view."""

    if not tls_metrics:
        return {
            "veh_total": 0.0,
            "mean_occupancy": 0.0,
            "max_occupancy": 0.0,
            "std_occupancy": 0.0,
            "mean_speed": 0.0,
            "critical_ratio": 0.0,
            "max_jam_len": 0.0
        }

    veh_totals = [float(m.get("veh_total", 0.0)) for m in tls_metrics]
    occupancies = [float(m.get("mean_occupancy", 0.0)) / 100 for m in tls_metrics]
    speeds = [float(m.get("mean_speed", 0.0)) for m in tls_metrics]
    max_jam_values = [float(m.get("max_jam", 0.0)) for m in tls_metrics]

    total_veh = sum(veh_totals)

    if total_veh > 0:
        mean_speed = sum(speed * veh for speed, veh in zip(speeds, veh_totals)) / total_veh
    else:
        mean_speed = 0.0

    # pandas.Series.std used ddof=1 in the old helper; keep that behavior when
    # possible, but avoid NaN for a single-TLS TAZ.
    occupancy_std = float(np.std(occupancies, ddof=1)) if len(occupancies) > 1 else 0.0

    return {
        "veh_total": total_veh,
        "mean_occupancy": sum(occupancies) / len(occupancies),
        "max_occupancy": max(occupancies),
        "std_occupancy": occupancy_std,
        "mean_speed": mean_speed,
        "critical_ratio": sum(o > 0.7 for o in occupancies) / len(occupancies),
        "max_jam_len": max(max_jam_values)
    }


class LocalTazTrafficEnv(EnvBase):
    """
    Local TAZ environment:
      - one agent per TAZ
      - one observation row per TAZ
      - one discrete action per TLS slot inside the TAZ
      - dense reward from absolute TAZ penalty plus local penalty improvement
      - terminal bonus aligned with a no-agent baseline on the same route/hour
    """

    ACTION_BIN_VALUES = (-5.0, 0.0, 5.0)
    ZERO_ACTION_INDEX = 1

    def __init__(
        self,
        sumoSimulator,
        stepSize: int = 300,
        device: str = "cpu",
        taz_map_path: str = TAZ_FILE,
        selected_taz_ids: Optional[list[str]] = None,
        speed_norm: float = 10.0,
        jam_norm: float = 10.0,
        warmupSteps: int = 0,
        cooldownSteps: int = 12,
        min_green: int = 25,
        max_green: int = 120,
        metric_clip: float = 2.0,
        reward_clip: float = 4.0,
        dense_abs_penalty_weight: float = 1.0,
        dense_delta_reward_weight: float = 0.40,
        terminal_bonus_weight: float = 1.0,
        comparison_reward_enabled: bool = True,
        waiting_reward_weight: float = 0.70,
        emission_reward_weight: float = 0.20,
        jam_reward_weight: float = 0.10,
        emission_co2_mix_weight: float = 0.50,
        emission_nox_mix_weight: float = 0.50,
        tls_waiting_time_ref: float = 1000.0,
        tls_co2_ref: float = 80000.0,
        tls_nox_ref: float = 40.0,
        tls_active_vehicle_ref: float = 80.0,
        taz_waiting_time_ref: float = 6000.0,
        taz_co2_ref: float = 300000.0,
        taz_nox_ref: float = 150.0,
        terminal_waiting_time_ref: float = 500000.0,
        terminal_co2_ref: float = 5000000000.0,
        terminal_nox_ref: float = 1800000.0,
        tls_veh_total_ref: float = 120.0,
        tls_pressure_ref: float = 10000.0,
        taz_veh_total_ref: float = 2500.0,
        taz_std_occupancy_ref: float = 0.25,
        max_jam_len_ref: float = 120.0,
        waiting_time_memory: int = 3600,
        action_signal_threshold: float = 0.03,
        sumo_seed: Optional[int] = None,
        sumo_thread_rngs: Optional[int] = None,
        tls_add_path: str = SUMO_NETWORK_PATH + "/optimized_tls.add.xml",
        **kwargs,
    ):
        super().__init__(device=device)

        self.sumo = sumoSimulator
        self.stepSize = int(stepSize)
        self.speed_norm = max(float(speed_norm), 1e-6)
        self.jam_norm = max(float(jam_norm), 1e-6)
        self.warmupSteps = int(warmupSteps)
        self.cooldownSteps = int(cooldownSteps)
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.metric_clip = max(float(metric_clip), 1.0)
        self.reward_clip = max(float(reward_clip), 1e-6)
        self.dense_abs_penalty_weight = float(dense_abs_penalty_weight)
        self.dense_delta_reward_weight = float(dense_delta_reward_weight)
        self.terminal_bonus_weight = float(terminal_bonus_weight)
        self.comparison_reward_enabled = bool(comparison_reward_enabled)
        self.waiting_reward_weight = float(waiting_reward_weight)
        self.emission_reward_weight = float(emission_reward_weight)
        self.jam_reward_weight = float(jam_reward_weight)
        self.emission_co2_mix_weight = float(emission_co2_mix_weight)
        self.emission_nox_mix_weight = float(emission_nox_mix_weight)
        self.tls_waiting_time_ref = max(float(tls_waiting_time_ref), 1e-6)
        self.tls_co2_ref = max(float(tls_co2_ref), 1e-6)
        self.tls_nox_ref = max(float(tls_nox_ref), 1e-6)
        self.tls_active_vehicle_ref = max(float(tls_active_vehicle_ref), 1e-6)
        self.taz_waiting_time_ref = max(float(taz_waiting_time_ref), 1e-6)
        self.taz_co2_ref = max(float(taz_co2_ref), 1e-6)
        self.taz_nox_ref = max(float(taz_nox_ref), 1e-6)
        self.terminal_waiting_time_ref = max(float(terminal_waiting_time_ref), 1e-6)
        self.terminal_co2_ref = max(float(terminal_co2_ref), 1e-6)
        self.terminal_nox_ref = max(float(terminal_nox_ref), 1e-6)
        self.tls_veh_total_ref = max(float(tls_veh_total_ref), 1e-6)
        self.tls_pressure_ref = max(float(tls_pressure_ref), 1e-6)
        self.taz_veh_total_ref = max(float(taz_veh_total_ref), 1e-6)
        self.taz_std_occupancy_ref = max(float(taz_std_occupancy_ref), 1e-6)
        self.max_jam_len_ref = max(float(max_jam_len_ref), 1e-6)
        self.waiting_time_memory = max(int(waiting_time_memory), 1)
        self.action_signal_threshold = max(float(action_signal_threshold), 0.0)
        self.sumo_seed = int(sumo_seed) if sumo_seed is not None else None
        self.sumo_thread_rngs = int(sumo_thread_rngs) if sumo_thread_rngs is not None else None
        self.tls_add_path = str(tls_add_path)
        self.waiting_ref = 60.0

        reward_weight_sum = (
            self.waiting_reward_weight
            + self.emission_reward_weight
            + self.jam_reward_weight
        )
        if reward_weight_sum <= 0.0:
            self.waiting_reward_weight = 0.70
            self.emission_reward_weight = 0.20
            self.jam_reward_weight = 0.10
            reward_weight_sum = 1.0
        self.waiting_reward_weight /= reward_weight_sum
        self.emission_reward_weight /= reward_weight_sum
        self.jam_reward_weight /= reward_weight_sum

        emission_mix_sum = self.emission_co2_mix_weight + self.emission_nox_mix_weight
        if emission_mix_sum <= 0.0:
            self.emission_co2_mix_weight = 0.50
            self.emission_nox_mix_weight = 0.50
            emission_mix_sum = 1.0
        self.emission_co2_mix_weight /= emission_mix_sum
        self.emission_nox_mix_weight /= emission_mix_sum

        self.taz_map_path = str(taz_map_path)
        self.selected_taz_ids = [str(taz_id) for taz_id in (selected_taz_ids or [])]
        raw_taz_tls_map = self._load_taz_tls_map(self.taz_map_path)
        self.taz_tls_map = self._filter_taz_tls_map(raw_taz_tls_map, self.selected_taz_ids)
        self.taz_ids = list(self.taz_tls_map.keys())
        self.tls_list = self._build_unique_tls_list(self.taz_tls_map)
        if len(self.tls_list) == 0:
            raise RuntimeError("No TLS found from TAZ mapping.")

        self.tls_to_taz = {}
        for taz in self.taz_ids:
            for tls in self.taz_tls_map[taz]:
                if tls not in self.tls_to_taz:
                    self.tls_to_taz[tls] = taz

        self.tls_num_phases, self.max_num_phases = self._compute_tls_phases_from_net(self.tls_add_path)
        self.max_num_phases = max(int(self.max_num_phases), 1)

        self.tls_by_taz = {taz: list(self.taz_tls_map[taz]) for taz in self.taz_ids}
        self.max_tls_per_taz = max(len(v) for v in self.tls_by_taz.values())
        self.per_tls_feature_dim = 1 + self.max_num_phases + 11
        self.per_taz_feature_dim = 14
        self.agent_obs_dim = self.max_tls_per_taz * self.per_tls_feature_dim + self.per_taz_feature_dim

        self.observation_spec = UnboundedSpec(
            shape=(len(self.taz_ids), self.agent_obs_dim),
            device=self.device,
        )
        self.action_spec = BoundedSpec(
            low=0,
            high=len(self.ACTION_BIN_VALUES) - 1,
            shape=(len(self.taz_ids), self.max_tls_per_taz),
            device=self.device,
        )
        self.reward_spec = UnboundedSpec(shape=(len(self.taz_ids),), device=self.device)

        action_mask = []
        for taz in self.taz_ids:
            valid = len(self.tls_by_taz[taz])
            row = [True] * valid + [False] * (self.max_tls_per_taz - valid)
            action_mask.append(row)
        self._action_mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        self._current_action_mask = self._action_mask.clone()
        self.action_bins = torch.tensor(self.ACTION_BIN_VALUES, dtype=torch.float32, device=self.device)

        self.current_hour = 0
        self.current_step = 0
        self.output_subdir = "output"
        self.tripinfo_filename = "tripinfos.xml"
        self.emission_filename = "emission-output.xml"

        self.reference_penalty = None
        self.reference_reward_components = {}
        self._prev_taz_penalties = np.zeros(len(self.taz_ids), dtype=np.float32)
        self._prev_action_taz_metrics = None
        self._action_signal_by_taz = {
            taz: 0.0
            for taz in self.taz_ids
        }
        self._episode_vehicle_waiting_by_id = {}
        self._episode_vehicle_taz_by_id = {}
        self._tls_lanes = {tls: set() for tls in self.tls_list}
        self._lane_to_tls_idx = {}
        self._edge_to_tls_idx = {}
        self._last_tls_metrics_by_id = {
            tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
            for tls in self.tls_list
        }
        self._last_taz_metrics_by_id = {
            taz: self._zero_taz_metrics()
            for taz in self.taz_ids
        }
        self._last_green_tls_snapshot = {}
        self._phase_duration_memory = {}
        self._active_program_id = {}
        self._episode_phase_initial_durations = {}
        self._episode_phase_cumulative_abs_deltas = {}
        self._episode_phase_signed_deltas = {}
        self._episode_phase_nonzero_updates = {}
        self._episode_phase_sign_flips = {}
        self._episode_phase_last_nonzero_sign = {}
        self._episode_duration_events = {}
        self._episode_max_jam_len = 0.0

        self.last_reward_components = {}
        self.last_duration_diagnostics = {}

    @staticmethod
    def _build_unique_tls_list(taz_tls_map: dict) -> list[str]:
        ordered = []
        seen = set()
        for taz in taz_tls_map:
            for tls in taz_tls_map[taz]:
                if tls in seen:
                    continue
                seen.add(tls)
                ordered.append(str(tls))
        return ordered

    def _load_taz_tls_map(self, taz_map_path: str) -> dict:
        sim_map = getattr(self.sumo, "tazTlsMap", None)
        if isinstance(sim_map, dict) and len(sim_map) > 0:
            return {
                str(k): [str(x) for x in v]
                for k, v in sim_map.items()
                if isinstance(v, list)
            }

        if not os.path.exists(taz_map_path):
            raise FileNotFoundError(f"TAZ map not found: {taz_map_path}")
        with open(taz_map_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise RuntimeError("Invalid TAZ map content: expected dict.")
        return {
            str(k): [str(x) for x in v]
            for k, v in raw.items()
            if isinstance(v, list)
        }

    @staticmethod
    def _filter_taz_tls_map(taz_tls_map: dict, selected_taz_ids: list[str]) -> dict:
        if not selected_taz_ids:
            return dict(taz_tls_map)

        requested = []
        seen = set()
        for taz_id in selected_taz_ids:
            key = str(taz_id)
            if key in seen:
                continue
            seen.add(key)
            requested.append(key)

        filtered = {
            taz_id: list(tls_ids)
            for taz_id, tls_ids in taz_tls_map.items()
            if str(taz_id) in seen
        }
        missing = [taz_id for taz_id in requested if taz_id not in filtered]
        if missing:
            available = ", ".join(sorted(str(k) for k in taz_tls_map.keys()))
            raise ValueError(
                f"Selected TAZ IDs not found: {missing}. Available TAZ IDs: {available}"
            )
        return filtered

    @staticmethod
    def _zero_tls_metrics(taz_id: Optional[str] = None) -> dict:
        return {
            "taz_id": str(taz_id) if taz_id is not None else None,
            "phase_id": 0,
            "phase_duration": 0.0,
            "veh_total": 0.0,
            "mean_speed": 0.0,
            "mean_occupancy": 0.0,
            "max_jam": 0.0,
            "pressure": 0.0,
            "active_vehicle_count": 0.0,
            "total_waiting_time": 0.0,
            "avg_waiting_time": 0.0,
            "total_co2": 0.0,
            "total_nox": 0.0,
        }

    @staticmethod
    def _zero_taz_metrics() -> dict:
        return {
            "mean_speed": 0.0,
            "max_jam_len": 0.0,
            "mean_occupancy": 0.0,
            "veh_total": 0.0,
            "critical_ratio": 0.0,
            "max_occupancy": 0.0,
            "std_occupancy": 0.0,
            "active_vehicle_count": 0.0,
            "total_waiting_time": 0.0,
            "avg_waiting_time": 0.0,
            "total_co2": 0.0,
            "total_nox": 0.0,
        }

    def set_episode_context(self, hour: int):
        self.current_hour = int(hour) % 24

    def set_baseline_penalty(self, penalty: Optional[float], components: Optional[dict] = None):
        if penalty is None:
            self.reference_penalty = None
            self.reference_reward_components = {}
            return
        self.reference_penalty = float(penalty)
        self.reference_reward_components = dict(components or {})

    def get_taz_ids(self) -> list[str]:
        return list(self.taz_ids)

    def get_tls_ids(self) -> list[str]:
        return list(self.tls_list)

    def get_action_mask(self) -> torch.Tensor:
        return self._current_action_mask.clone()

    def decode_action_indices(self, action_index: torch.Tensor) -> torch.Tensor:
        return self.action_bins[action_index.long().clamp(0, len(self.ACTION_BIN_VALUES) - 1)]

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except Exception:
            return default
        if not np.isfinite(parsed):
            return default
        return parsed

    @staticmethod
    def _get_attr_case_insensitive(elem: ET.Element, key: str):
        if key in elem.attrib:
            return elem.attrib[key]
        target = key.lower()
        for existing_key, existing_value in elem.attrib.items():
            if existing_key.lower() == target:
                return existing_value
        return None

    def _safe_parse_xml_root(self, xml_path: str):
        try:
            return ET.parse(xml_path).getroot()
        except Exception:
            pass

        try:
            with open(xml_path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except Exception:
            return None
        if text is None:
            return None

        text = text.replace("\ufeff", "")
        text = re.sub(r"<\?xml[^>]*\?>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = text.strip()
        if not text:
            return None

        root_tag = None
        for match in re.finditer(r"<([A-Za-z_][\w\-\.:]*)[^>]*>", text):
            tag = match.group(1)
            if tag.startswith("?") or tag.startswith("!"):
                continue
            if tag.lower() == "xml":
                continue
            root_tag = tag
            break
        if root_tag is None:
            return None

        closing = f"</{root_tag}>"
        close_pos = text.rfind(closing)
        if close_pos >= 0:
            candidate = text[: close_pos + len(closing)]
        else:
            last_gt = text.rfind(">")
            if last_gt <= 0:
                return None
            candidate = text[: last_gt + 1] + closing

        for _ in range(32):
            try:
                return ET.fromstring(candidate)
            except Exception:
                cut = candidate.rfind("<", 0, max(0, len(candidate) - len(closing)))
                if cut <= 0:
                    break
                candidate = candidate[:cut] + closing
        return None

    @staticmethod
    def _lane_to_edge_id(lane_id: str) -> str:
        lane_id = str(lane_id or "")
        if not lane_id:
            return ""
        if lane_id.startswith(":"):
            return lane_id
        if "_" in lane_id:
            return lane_id.rsplit("_", 1)[0]
        return lane_id

    @staticmethod
    def _is_green_phase_state(phase_state: str) -> bool:
        return ("g" in phase_state) or ("G" in phase_state)

    @staticmethod
    def _has_yellow_phase_state(phase_state: str) -> bool:
        return ("y" in phase_state) or ("Y" in phase_state)

    @classmethod
    def _is_editable_phase_state(cls, phase_state: str) -> bool:
        return cls._is_green_phase_state(phase_state) and not cls._has_yellow_phase_state(phase_state)

    @staticmethod
    def _normalize_phase_duration_value(duration: float, fallback: float = 1.0) -> int:
        try:
            value = float(duration)
        except Exception:
            value = float(fallback)
        if not np.isfinite(value):
            value = float(fallback)
        return max(int(round(value)), 1)

    def _get_time_features(self):
        angle = 2.0 * np.pi * (self.current_hour % 24) / 24.0
        return float(np.sin(angle)), float(np.cos(angle))

    def _compute_tls_phases_from_net(self, net_path: str):
        tls_num_phases = {}
        max_phases = 0
        try:
            tree = ET.parse(net_path)
            root = tree.getroot()
            for tl_logic in root.findall("tlLogic"):
                tls_id = str(tl_logic.get("id"))
                if tls_id not in self.tls_list:
                    continue
                n_phases = len(tl_logic.findall("phase"))
                if n_phases <= 0:
                    n_phases = 1
                tls_num_phases[tls_id] = int(n_phases)
                max_phases = max(max_phases, int(n_phases))
        except Exception:
            pass

        if max_phases <= 0:
            for tls in self.tls_list:
                tls_num_phases[tls] = 1
            max_phases = 1
        else:
            for tls in self.tls_list:
                tls_num_phases.setdefault(tls, 1)
        return tls_num_phases, max_phases

    def _find_previous_editable_phase_id(self, program, current_phase_id: int) -> Optional[int]:
        phases = list(getattr(program, "phases", []))
        n_phases = len(phases)
        if n_phases == 0:
            return None

        idx = (int(current_phase_id) - 1) % n_phases
        for _ in range(n_phases):
            if self._is_editable_phase_state(getattr(phases[idx], "state", "")):
                return idx
            idx = (idx - 1) % n_phases
        return None

    def _get_active_program_logic(self, tls_id: str):
        all_programs = list(libtraci.trafficlight.getAllProgramLogics(tls_id))
        if not all_programs:
            return None, None
        active_program_id = str(libtraci.trafficlight.getProgram(tls_id))
        for prog in all_programs:
            if str(getattr(prog, "programID", "")) == active_program_id:
                return prog, active_program_id
        return all_programs[0], str(getattr(all_programs[0], "programID", ""))

    def _sync_phase_duration_memory(self, tls_id: str):
        try:
            program, program_id = self._get_active_program_logic(tls_id)
        except Exception:
            return
        if program is None:
            return
        durations = [self._safe_float(getattr(ph, "duration", 0.0), 0.0) for ph in list(program.phases)]
        if not durations:
            return
        self._phase_duration_memory[tls_id] = durations
        self._active_program_id[tls_id] = program_id

    def _get_phase_spent_duration(self, tls_id: str) -> float:
        try:
            spent = self._safe_float(libtraci.trafficlight.getSpentDuration(tls_id), 0.0)
            if spent >= 0.0:
                return float(spent)
        except Exception:
            pass

        # Fallback when direct spent-duration API is unavailable.
        try:
            next_switch = self._safe_float(libtraci.trafficlight.getNextSwitch(tls_id), np.nan)
            sim_time = self._safe_float(libtraci.simulation.getTime(), np.nan)
            total_duration = self._safe_float(libtraci.trafficlight.getPhaseDuration(tls_id), np.nan)
            if np.isfinite(next_switch) and np.isfinite(sim_time) and np.isfinite(total_duration):
                remaining = max(next_switch - sim_time, 0.0)
                return float(max(total_duration - remaining, 0.0))
        except Exception:
            pass
        return 0.0

    def _reset_duration_diagnostics(self):
        self._episode_phase_initial_durations = {}
        self._episode_phase_cumulative_abs_deltas = {}
        self._episode_phase_signed_deltas = {}
        self._episode_phase_nonzero_updates = {}
        self._episode_phase_sign_flips = {}
        self._episode_phase_last_nonzero_sign = {}
        self._episode_duration_events = {}

        for tls_id, durations in self._phase_duration_memory.items():
            phase_count = len(durations)
            self._episode_phase_initial_durations[tls_id] = [float(x) for x in durations]
            self._episode_phase_cumulative_abs_deltas[tls_id] = [0.0 for _ in range(phase_count)]
            self._episode_phase_signed_deltas[tls_id] = [0.0 for _ in range(phase_count)]
            self._episode_phase_nonzero_updates[tls_id] = [0 for _ in range(phase_count)]
            self._episode_phase_sign_flips[tls_id] = [0 for _ in range(phase_count)]
            self._episode_phase_last_nonzero_sign[tls_id] = [0 for _ in range(phase_count)]
            self._episode_duration_events[tls_id] = []

    def _record_duration_event(
        self,
        tls_id: str,
        target_phase_id: int,
        requested_delta: float,
        base_duration: float,
        applied_duration: float,
    ):
        initial = self._episode_phase_initial_durations.get(tls_id)
        cumulative = self._episode_phase_cumulative_abs_deltas.get(tls_id)
        signed = self._episode_phase_signed_deltas.get(tls_id)
        nonzero = self._episode_phase_nonzero_updates.get(tls_id)
        flips = self._episode_phase_sign_flips.get(tls_id)
        last_signs = self._episode_phase_last_nonzero_sign.get(tls_id)
        if (
            initial is None
            or cumulative is None
            or signed is None
            or nonzero is None
            or flips is None
            or last_signs is None
            or target_phase_id < 0
            or target_phase_id >= len(initial)
        ):
            return

        applied_delta = float(applied_duration - base_duration)
        self._episode_duration_events.setdefault(tls_id, []).append(
            {
                "control_step": int(self.current_step),
                "phase_id": int(target_phase_id),
                "requested_delta": float(requested_delta),
                "applied_delta": float(applied_delta),
                "base_duration": float(base_duration),
                "applied_duration": float(applied_duration),
            }
        )

        if abs(applied_delta) <= 1e-9:
            return

        cumulative[target_phase_id] += abs(applied_delta)
        signed[target_phase_id] += applied_delta
        nonzero[target_phase_id] += 1

        sign = 1 if applied_delta > 0.0 else -1
        prev_sign = int(last_signs[target_phase_id])
        if prev_sign != 0 and prev_sign != sign:
            flips[target_phase_id] += 1
        last_signs[target_phase_id] = sign

    def get_episode_duration_diagnostics(self) -> dict:
        per_tls = {}
        summary_net_abs = []
        summary_cumulative_abs = []
        summary_ratio = []
        summary_sign_flips = []
        summary_nonzero_updates = []
        reverted_count = 0
        changed_count = 0

        for tls_id in self.tls_list:
            initial = [float(x) for x in self._episode_phase_initial_durations.get(tls_id, [])]
            final = [float(x) for x in self._phase_duration_memory.get(tls_id, initial)]
            phase_count = min(len(initial), len(final))
            if len(initial) != phase_count:
                initial = initial[:phase_count]
            if len(final) != phase_count:
                final = final[:phase_count]

            net_deltas = [float(f - i) for i, f in zip(initial, final)]
            cumulative_abs = [
                float(x)
                for x in self._episode_phase_cumulative_abs_deltas.get(tls_id, [0.0 for _ in range(phase_count)])
            ][:phase_count]
            signed_deltas = [
                float(x)
                for x in self._episode_phase_signed_deltas.get(tls_id, [0.0 for _ in range(phase_count)])
            ][:phase_count]
            nonzero_updates = [
                int(x)
                for x in self._episode_phase_nonzero_updates.get(tls_id, [0 for _ in range(phase_count)])
            ][:phase_count]
            sign_flips = [
                int(x)
                for x in self._episode_phase_sign_flips.get(tls_id, [0 for _ in range(phase_count)])
            ][:phase_count]

            total_net_abs = float(sum(abs(x) for x in net_deltas))
            total_cumulative_abs = float(sum(cumulative_abs))
            net_to_cumulative_ratio = (
                float(total_net_abs / total_cumulative_abs)
                if total_cumulative_abs > 1e-9 else 0.0
            )
            total_sign_flips = int(sum(sign_flips))
            total_nonzero_updates = int(sum(nonzero_updates))
            reverted_to_initial = bool(total_net_abs <= 1e-9)
            changed = bool(total_cumulative_abs > 1e-9)

            if reverted_to_initial:
                reverted_count += 1
            if changed:
                changed_count += 1

            summary_net_abs.append(total_net_abs)
            summary_cumulative_abs.append(total_cumulative_abs)
            summary_ratio.append(net_to_cumulative_ratio)
            summary_sign_flips.append(float(total_sign_flips))
            summary_nonzero_updates.append(float(total_nonzero_updates))

            per_tls[tls_id] = {
                "initial_phase_durations": initial,
                "final_phase_durations": final,
                "phase_net_deltas": net_deltas,
                "phase_signed_applied_deltas": signed_deltas,
                "phase_cumulative_abs_deltas": cumulative_abs,
                "phase_nonzero_updates": nonzero_updates,
                "phase_sign_flips": sign_flips,
                "total_net_abs_delta": total_net_abs,
                "total_cumulative_abs_delta": total_cumulative_abs,
                "net_to_cumulative_ratio": float(net_to_cumulative_ratio),
                "total_sign_flips": total_sign_flips,
                "total_nonzero_updates": total_nonzero_updates,
                "reverted_to_initial": reverted_to_initial,
                "changed": changed,
                "events": list(self._episode_duration_events.get(tls_id, [])),
            }

        tls_count = max(len(self.tls_list), 1)
        return {
            "summary": {
                "tls_count": int(len(self.tls_list)),
                "mean_total_net_abs_delta": float(sum(summary_net_abs) / tls_count) if summary_net_abs else 0.0,
                "mean_total_cumulative_abs_delta": float(sum(summary_cumulative_abs) / tls_count) if summary_cumulative_abs else 0.0,
                "mean_net_to_cumulative_ratio": float(sum(summary_ratio) / tls_count) if summary_ratio else 0.0,
                "mean_total_sign_flips": float(sum(summary_sign_flips) / tls_count) if summary_sign_flips else 0.0,
                "mean_total_nonzero_updates": float(sum(summary_nonzero_updates) / tls_count) if summary_nonzero_updates else 0.0,
                "reverted_tls_ratio": float(reverted_count / tls_count),
                "changed_tls_ratio": float(changed_count / tls_count),
            },
            "per_tls": per_tls,
        }

    def _build_tls_lane_cache(self):
        self._tls_lanes = {tls: set() for tls in self.tls_list}
        self._lane_to_tls_idx = {}
        self._edge_to_tls_idx = {}

        for tls_idx, tls in enumerate(self.tls_list):
            try:
                tls_lanes = set(self.sumo.get_tls_lanes(tls))
            except Exception:
                tls_lanes = set()
            self._tls_lanes[tls] = tls_lanes

            for lane_id in tls_lanes:
                lane_id = str(lane_id)
                if lane_id not in self._lane_to_tls_idx:
                    self._lane_to_tls_idx[lane_id] = int(tls_idx)
                edge_id = self._lane_to_edge_id(lane_id)
                if edge_id and edge_id not in self._edge_to_tls_idx:
                    self._edge_to_tls_idx[edge_id] = int(tls_idx)

    def _apply_action_to_tls(self, tls_id: str, action_value: float) -> float:
        try:
            program, program_id = self._get_active_program_logic(tls_id)
        except Exception:
            return 0.0
        if program is None:
            return 0.0

        if (
            tls_id not in self._phase_duration_memory
            or tls_id not in self._active_program_id
            or self._active_program_id.get(tls_id) != program_id
            or len(self._phase_duration_memory.get(tls_id, [])) != len(list(program.phases))
        ):
            self._sync_phase_duration_memory(tls_id)

        phase_memory = self._phase_duration_memory.get(tls_id, [])
        if len(phase_memory) != len(list(program.phases)):
            phase_memory = [self._safe_float(getattr(ph, "duration", 0.0), 0.0) for ph in list(program.phases)]
            self._phase_duration_memory[tls_id] = phase_memory

        current_phase_id = int(program.currentPhaseIndex)
        target_phase_id = current_phase_id
        current_phase = program.phases[current_phase_id]
        spent_duration = 0.0
        if not self._is_editable_phase_state(current_phase.state):
            prev_editable_phase_id = self._find_previous_editable_phase_id(program, current_phase_id)
            if prev_editable_phase_id is None:
                return 0.0
            target_phase_id = int(prev_editable_phase_id)
        else:
            spent_duration = self._get_phase_spent_duration(tls_id)

        base = self._safe_float(
            phase_memory[target_phase_id],
            self._safe_float(program.phases[target_phase_id].duration, 0.0),
        )
        new_dur = int(np.clip(base + float(action_value), self.min_green, self.max_green))
        self._phase_duration_memory[tls_id][target_phase_id] = float(new_dur)
        remaining_after_update = max(float(new_dur) - float(spent_duration), 0.0)
        remaining_after_update = float(
            self._normalize_phase_duration_value(remaining_after_update, fallback=1.0)
        )

        applied_duration = float(new_dur)
        try:
            for idx, phase in enumerate(list(program.phases)):
                stored_duration = self._safe_float(
                    self._phase_duration_memory[tls_id][idx],
                    self._safe_float(getattr(phase, "duration", 0.0), 0.0),
                )
                if idx == target_phase_id and self._is_editable_phase_state(getattr(phase, "state", "")):
                    duration_value = self._normalize_phase_duration_value(new_dur, fallback=stored_duration)
                else:
                    duration_value = self._normalize_phase_duration_value(
                        stored_duration,
                        fallback=self._safe_float(getattr(phase, "duration", 0.0), 1.0),
                    )
                phase.maxDur = duration_value
                phase.minDur = duration_value
                phase.duration = duration_value
            try:
                program.currentPhaseIndex = current_phase_id
            except Exception:
                pass
            libtraci.trafficlight.setProgramLogic(tls_id, program)

            if target_phase_id == current_phase_id and self._is_editable_phase_state(current_phase.state):
                try:
                    # Apply only the remaining time so the current phase does not restart from zero.
                    libtraci.trafficlight.setPhaseDuration(tls_id, remaining_after_update)
                except Exception:
                    pass

            after_program, _ = self._get_active_program_logic(tls_id)
            if after_program is not None:
                self._sync_phase_duration_memory(tls_id)
                synced_memory = self._phase_duration_memory.get(tls_id, [])
                applied_duration = self._safe_float(
                    synced_memory[target_phase_id] if target_phase_id < len(synced_memory) else float(new_dur),
                    self._safe_float(after_program.phases[target_phase_id].duration, float(new_dur)),
                )
        except Exception:
            try:
                self.sumo.set_tls_phase_duration(tls_id, target_phase_id, new_dur)
            except Exception:
                pass

        self._last_green_tls_snapshot[tls_id] = {
            **self._last_green_tls_snapshot.get(tls_id, {}),
            "phase_id": target_phase_id,
            "phase_duration": float(applied_duration),
        }
        self._record_duration_event(
            tls_id=tls_id,
            target_phase_id=target_phase_id,
            requested_delta=float(action_value),
            base_duration=float(base),
            applied_duration=float(applied_duration),
        )
        return float(applied_duration - base)

    def _collect_vehicle_metrics_by_tls(self) -> dict:
        metrics_by_tls = {
            tls: {
                "active_vehicle_count": 0.0,
                "total_waiting_time": 0.0,
                "avg_waiting_time": 0.0,
                "total_co2": 0.0,
                "total_nox": 0.0,
            }
            for tls in self.tls_list
        }
        if not self.sumo.isRunning():
            return metrics_by_tls

        try:
            vehicle_ids = list(libtraci.vehicle.getIDList())
        except Exception:
            return metrics_by_tls

        for vehicle_id in vehicle_ids:
            try:
                lane_id = str(libtraci.vehicle.getLaneID(vehicle_id))
            except Exception:
                lane_id = ""
            tls_idx = self._lane_to_tls_idx.get(lane_id)

            if tls_idx is None:
                edge_id = self._lane_to_edge_id(lane_id)
                tls_idx = self._edge_to_tls_idx.get(edge_id)
            if tls_idx is None:
                try:
                    road_id = str(libtraci.vehicle.getRoadID(vehicle_id))
                except Exception:
                    road_id = ""
                tls_idx = self._edge_to_tls_idx.get(road_id)
            if tls_idx is None:
                continue

            try:
                # SUMO accumulated waiting time depends on --waiting-time-memory.
                waiting_time = self._safe_float(libtraci.vehicle.getAccumulatedWaitingTime(vehicle_id), 0.0)
                co2 = self._safe_float(libtraci.vehicle.getCO2Emission(vehicle_id), 0.0)
                nox = self._safe_float(libtraci.vehicle.getNOxEmission(vehicle_id), 0.0)
            except Exception:
                continue

            tls_id = self.tls_list[int(tls_idx)]
            taz_id = self.tls_to_taz.get(tls_id)
            prev_waiting = self._safe_float(self._episode_vehicle_waiting_by_id.get(vehicle_id, 0.0), 0.0)
            self._episode_vehicle_waiting_by_id[vehicle_id] = float(max(prev_waiting, waiting_time))
            if taz_id is not None:
                self._episode_vehicle_taz_by_id[vehicle_id] = str(taz_id)
            metrics_by_tls[tls_id]["active_vehicle_count"] += 1.0
            metrics_by_tls[tls_id]["total_waiting_time"] += float(waiting_time)
            metrics_by_tls[tls_id]["total_co2"] += float(co2)
            metrics_by_tls[tls_id]["total_nox"] += float(nox)

        for tls_id in self.tls_list:
            count = metrics_by_tls[tls_id]["active_vehicle_count"]
            if count > 0:
                metrics_by_tls[tls_id]["avg_waiting_time"] = (
                    metrics_by_tls[tls_id]["total_waiting_time"] / float(count)
                )
        return metrics_by_tls

    def _collect_tls_runtime_metrics(self, tls_id: str, aggregated: dict, vehicle_metrics: dict) -> dict:
        try:
            program = libtraci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase_id = int(program.currentPhaseIndex)
            current_phase = program.phases[current_phase_id]
            current_phase_state = getattr(current_phase, "state", "")
        except Exception:
            program = None
            current_phase_id = int(libtraci.trafficlight.getPhase(tls_id))
            current_phase_state = ""

        try:
            current_phase_duration = float(libtraci.trafficlight.getPhaseDuration(tls_id))
        except Exception:
            if program is not None and 0 <= current_phase_id < len(program.phases):
                current_phase_duration = self._safe_float(program.phases[current_phase_id].duration, 0.0)
            else:
                current_phase_duration = 0.0

        phase_id_for_obs = current_phase_id
        phase_duration_for_obs = current_phase_duration
        metrics_for_obs = {
            "veh_total": self._safe_float(aggregated.get("veh_total", 0.0), 0.0),
            "mean_speed": self._safe_float(aggregated.get("mean_speed", 0.0), 0.0),
            "mean_occupancy": self._safe_float(aggregated.get("mean_occupancy", 0.0), 0.0),
            "max_jam": self._safe_float(aggregated.get("max_jam", 0.0), 0.0),
            "pressure": self._safe_float(aggregated.get("pressure", 0.0), 0.0),
            "active_vehicle_count": self._safe_float(vehicle_metrics.get("active_vehicle_count", 0.0), 0.0),
            "total_waiting_time": self._safe_float(vehicle_metrics.get("total_waiting_time", 0.0), 0.0),
            "avg_waiting_time": self._safe_float(vehicle_metrics.get("avg_waiting_time", 0.0), 0.0),
            "total_co2": self._safe_float(vehicle_metrics.get("total_co2", 0.0), 0.0),
            "total_nox": self._safe_float(vehicle_metrics.get("total_nox", 0.0), 0.0),
        }

        if self._is_editable_phase_state(current_phase_state):
            self._last_green_tls_snapshot[tls_id] = {
                "phase_id": int(phase_id_for_obs),
                "phase_duration": float(phase_duration_for_obs),
                **metrics_for_obs,
            }
        else:
            if program is not None:
                prev_editable_phase_id = self._find_previous_editable_phase_id(program, current_phase_id)
                if prev_editable_phase_id is not None:
                    phase_id_for_obs = int(prev_editable_phase_id)
                    phase_duration_for_obs = self._safe_float(
                        program.phases[prev_editable_phase_id].duration,
                        phase_duration_for_obs,
                    )

            cached_metrics = self._last_green_tls_snapshot.get(tls_id)
            if cached_metrics is not None:
                metrics_for_obs = {
                    "veh_total": self._safe_float(cached_metrics.get("veh_total", metrics_for_obs["veh_total"]), 0.0),
                    "mean_speed": self._safe_float(cached_metrics.get("mean_speed", metrics_for_obs["mean_speed"]), 0.0),
                    "mean_occupancy": self._safe_float(cached_metrics.get("mean_occupancy", metrics_for_obs["mean_occupancy"]), 0.0),
                    "max_jam": self._safe_float(cached_metrics.get("max_jam", metrics_for_obs["max_jam"]), 0.0),
                    "pressure": self._safe_float(cached_metrics.get("pressure", metrics_for_obs["pressure"]), 0.0),
                    "active_vehicle_count": self._safe_float(cached_metrics.get("active_vehicle_count", metrics_for_obs["active_vehicle_count"]), 0.0),
                    "total_waiting_time": self._safe_float(cached_metrics.get("total_waiting_time", metrics_for_obs["total_waiting_time"]), 0.0),
                    "avg_waiting_time": self._safe_float(cached_metrics.get("avg_waiting_time", metrics_for_obs["avg_waiting_time"]), 0.0),
                    "total_co2": self._safe_float(cached_metrics.get("total_co2", metrics_for_obs["total_co2"]), 0.0),
                    "total_nox": self._safe_float(cached_metrics.get("total_nox", metrics_for_obs["total_nox"]), 0.0),
                }
                phase_duration_for_obs = self._safe_float(
                    cached_metrics.get("phase_duration", phase_duration_for_obs),
                    phase_duration_for_obs,
                )

        return {
            "taz_id": self.tls_to_taz.get(tls_id),
            "tls_id": tls_id,
            "phase_id": int(phase_id_for_obs),
            "phase_duration": float(phase_duration_for_obs),
            **metrics_for_obs,
        }

    def _build_taz_metrics_from_tls(self) -> dict:
        metrics_by_taz = {
            taz: self._zero_taz_metrics()
            for taz in self.taz_ids
        }

        for taz in self.taz_ids:
            tls_metrics = [
                self._last_tls_metrics_by_id.get(
                    tls,
                    self._zero_tls_metrics(taz_id=taz),
                )
                for tls in self.tls_by_taz[taz]
            ]
            detector_aggregate = aggregate_taz_metrics(tls_metrics)
            active_vehicle_count = float(sum(m.get("active_vehicle_count", 0.0) for m in tls_metrics))
            total_waiting_time = float(sum(m.get("total_waiting_time", 0.0) for m in tls_metrics))
            total_co2 = float(sum(m.get("total_co2", 0.0) for m in tls_metrics))
            total_nox = float(sum(m.get("total_nox", 0.0) for m in tls_metrics))
            avg_waiting_time = total_waiting_time / active_vehicle_count if active_vehicle_count > 0.0 else 0.0

            metrics_by_taz[taz] = {
                "mean_speed": self._safe_float(detector_aggregate.get("mean_speed", 0.0), 0.0),
                "max_jam_len": self._safe_float(detector_aggregate.get("max_jam_len", 0.0), 0.0),
                "mean_occupancy": self._safe_float(detector_aggregate.get("mean_occupancy", 0.0), 0.0),
                "veh_total": self._safe_float(detector_aggregate.get("veh_total", 0.0), 0.0),
                "critical_ratio": self._safe_float(detector_aggregate.get("critical_ratio", 0.0), 0.0),
                "max_occupancy": self._safe_float(detector_aggregate.get("max_occupancy", 0.0), 0.0),
                "std_occupancy": self._safe_float(detector_aggregate.get("std_occupancy", 0.0), 0.0),
                "active_vehicle_count": active_vehicle_count,
                "total_waiting_time": total_waiting_time,
                "avg_waiting_time": avg_waiting_time,
                "total_co2": total_co2,
                "total_nox": total_nox,
            }
        return metrics_by_taz

    def _get_episode_waiting_metrics(self) -> dict:
        totals_by_taz = {
            taz: 0.0
            for taz in self.taz_ids
        }
        counts_by_taz = {
            taz: 0
            for taz in self.taz_ids
        }

        for vehicle_id, waiting_time in self._episode_vehicle_waiting_by_id.items():
            taz_id = self._episode_vehicle_taz_by_id.get(vehicle_id)
            if taz_id not in totals_by_taz:
                continue
            totals_by_taz[taz_id] += float(waiting_time)
            counts_by_taz[taz_id] += 1

        total_waiting_time = float(sum(totals_by_taz.values()))
        vehicle_count = int(sum(counts_by_taz.values()))
        avg_waiting_time = total_waiting_time / float(vehicle_count) if vehicle_count > 0 else 0.0
        return {
            "vehicle_count": int(vehicle_count),
            "avg_waiting_time": float(avg_waiting_time),
            "total_waiting_time": float(total_waiting_time),
            "waiting_time_by_taz": {
                taz: float(totals_by_taz[taz])
                for taz in self.taz_ids
            },
            "vehicle_count_by_taz": {
                taz: int(counts_by_taz[taz])
                for taz in self.taz_ids
            },
        }

    def _clone_taz_metrics_snapshot(self) -> dict:
        return {
            taz: {
                key: self._safe_float(value, 0.0)
                for key, value in metrics.items()
            }
            for taz, metrics in self._last_taz_metrics_by_id.items()
        }

    def _build_runtime_action_mask(self) -> torch.Tensor:
        runtime_mask = self._action_mask.clone()
        self._action_signal_by_taz = {
            taz: 0.0
            for taz in self.taz_ids
        }
        if self._prev_action_taz_metrics is None:
            # Allow control from the first decision step instead of forcing a no-op rollout prefix.
            return runtime_mask

        for taz_idx, taz in enumerate(self.taz_ids):
            current = self._last_taz_metrics_by_id.get(taz, self._zero_taz_metrics())
            previous = self._prev_action_taz_metrics.get(taz, self._zero_taz_metrics())

            waiting_delta = abs(
                self._safe_float(current.get("total_waiting_time", 0.0), 0.0)
                - self._safe_float(previous.get("total_waiting_time", 0.0), 0.0)
            ) / self.taz_waiting_time_ref
            active_delta = abs(
                self._safe_float(current.get("active_vehicle_count", 0.0), 0.0)
                - self._safe_float(previous.get("active_vehicle_count", 0.0), 0.0)
            ) / self.taz_veh_total_ref
            jam_delta = abs(
                self._safe_float(current.get("max_jam_len", 0.0), 0.0)
                - self._safe_float(previous.get("max_jam_len", 0.0), 0.0)
            ) / self.max_jam_len_ref
            speed_delta = abs(
                self._safe_float(current.get("mean_speed", 0.0), 0.0)
                - self._safe_float(previous.get("mean_speed", 0.0), 0.0)
            ) / self.speed_norm

            signal = waiting_delta + 0.5 * active_delta + 0.5 * jam_delta + 0.25 * speed_delta
            self._action_signal_by_taz[taz] = float(signal)
            if signal < self.action_signal_threshold:
                runtime_mask[taz_idx, :] = False
        return runtime_mask

    def _build_taz_penalties(self):
        penalties = []
        details = {}
        for taz in self.taz_ids:
            metrics = self._last_taz_metrics_by_id.get(taz, self._zero_taz_metrics())
            max_jam_len = self._safe_float(metrics.get("max_jam_len", 0.0), 0.0)
            total_waiting_time = self._safe_float(metrics.get("total_waiting_time", 0.0), 0.0)
            avg_waiting_time = self._safe_float(metrics.get("avg_waiting_time", 0.0), 0.0)
            total_co2 = self._safe_float(metrics.get("total_co2", 0.0), 0.0)
            total_nox = self._safe_float(metrics.get("total_nox", 0.0), 0.0)

            waiting_penalty = float(np.clip(total_waiting_time / self.taz_waiting_time_ref, 0.0, self.metric_clip))
            co2_penalty = float(np.clip(total_co2 / self.taz_co2_ref, 0.0, self.metric_clip))
            nox_penalty = float(np.clip(total_nox / self.taz_nox_ref, 0.0, self.metric_clip))
            emission_penalty = (
                self.emission_co2_mix_weight * co2_penalty
                + self.emission_nox_mix_weight * nox_penalty
            )
            jam_penalty = float(np.clip(max_jam_len / self.max_jam_len_ref, 0.0, self.metric_clip))

            penalty = (
                self.waiting_reward_weight * waiting_penalty
                + self.emission_reward_weight * emission_penalty
                + self.jam_reward_weight * jam_penalty
            )
            penalties.append(float(penalty))
            details[taz] = {
                "penalty": float(penalty),
                "waiting_penalty": float(waiting_penalty),
                "emission_penalty": float(emission_penalty),
                "co2_penalty": float(co2_penalty),
                "nox_penalty": float(nox_penalty),
                "jam_penalty": float(jam_penalty),
                "active_vehicle_count": float(self._safe_float(metrics.get("active_vehicle_count", 0.0), 0.0)),
                "avg_waiting_time": float(avg_waiting_time),
                "total_waiting_time": float(total_waiting_time),
                "total_co2": float(total_co2),
                "total_nox": float(total_nox),
                "max_jam_len": float(max_jam_len),
            }
        return np.asarray(penalties, dtype=np.float32), details

    def _compute_dense_reward_vector(self, penalties: np.ndarray) -> np.ndarray:
        delta = self._prev_taz_penalties - penalties
        reward_vector = (
            -self.dense_abs_penalty_weight * penalties
            + self.dense_delta_reward_weight * delta
        )
        return np.clip(reward_vector, -self.reward_clip, self.reward_clip).astype(np.float32)

    def _collect_observation(self) -> torch.Tensor:
        if not self.sumo.isRunning():
            self._last_tls_metrics_by_id = {
                tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
                for tls in self.tls_list
            }
            self._last_taz_metrics_by_id = {
                taz: self._zero_taz_metrics()
                for taz in self.taz_ids
            }
            self._current_action_mask = torch.zeros_like(self._action_mask, dtype=torch.bool, device=self.device)
            return torch.zeros(self.observation_spec.shape, dtype=torch.float32, device=self.device)

        raw_by_taz = {}
        for taz in self.taz_ids:
            try:
                raw_by_taz[taz] = self.sumo.get_taz_e2_metrics(taz, interval="last", mode="dict")
            except Exception:
                raw_by_taz[taz] = {"tls_data": {}, "taz_avg": {}}

        vehicle_metrics_by_tls = self._collect_vehicle_metrics_by_tls()
        tls_metrics_by_id = {}
        for taz in self.taz_ids:
            tls_data = raw_by_taz[taz].get("tls_data", {})
            for tls in self.tls_by_taz[taz]:
                aggregated = aggregate_tls_metrics(tls_data.get(tls, []))
                tls_metrics_by_id[tls] = self._collect_tls_runtime_metrics(
                    tls_id=tls,
                    aggregated=aggregated,
                    vehicle_metrics=vehicle_metrics_by_tls.get(tls, {}),
                )

        for tls in self.tls_list:
            tls_metrics_by_id.setdefault(
                tls,
                self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls)),
            )
        self._last_tls_metrics_by_id = tls_metrics_by_id
        self._last_taz_metrics_by_id = self._build_taz_metrics_from_tls()

        obs_rows = []
        hour_sin, hour_cos = self._get_time_features()
        for taz in self.taz_ids:
            row = []
            tls_ids = self.tls_by_taz[taz]
            for slot_idx in range(self.max_tls_per_taz):
                if slot_idx < len(tls_ids):
                    tls = tls_ids[slot_idx]
                    metrics = tls_metrics_by_id[tls]
                    phase_id = int(metrics.get("phase_id", 0))
                    n_phases = int(self.tls_num_phases.get(tls, self.max_num_phases))
                    one_hot = np.zeros(self.max_num_phases, dtype=np.float32)
                    if 0 <= phase_id < n_phases:
                        one_hot[phase_id] = 1.0

                    mean_occ = self._safe_float(metrics.get("mean_occupancy", 0.0), 0.0)
                    occ_norm = mean_occ / 100.0 if mean_occ > 1.5 else mean_occ
                    occ_norm = self._clip01(occ_norm)

                    row.extend(
                        [1.0]
                        + one_hot.tolist()
                        + [
                            float(np.clip(self._safe_float(metrics.get("phase_duration", 0.0), 0.0) / float(self.max_green), 0.0, 2.0)),
                            float(np.clip(self._safe_float(metrics.get("mean_speed", 0.0), 0.0) / self.speed_norm, 0.0, 2.0)),
                            float(np.clip(self._safe_float(metrics.get("max_jam", 0.0), 0.0) / self.jam_norm, 0.0, 2.0)),
                            float(occ_norm),
                            float(np.clip(self._safe_float(metrics.get("veh_total", 0.0), 0.0) / self.tls_veh_total_ref, 0.0, 5.0)),
                            float(np.clip(max(self._safe_float(metrics.get("pressure", 0.0), 0.0), 0.0) / self.tls_pressure_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("active_vehicle_count", 0.0), 0.0) / self.tls_active_vehicle_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("total_waiting_time", 0.0), 0.0) / self.tls_waiting_time_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("avg_waiting_time", 0.0), 0.0) / self.waiting_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("total_co2", 0.0), 0.0) / self.tls_co2_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("total_nox", 0.0), 0.0) / self.tls_nox_ref, 0.0, 5.0)),
                        ]
                    )
                else:
                    row.extend([0.0] * self.per_tls_feature_dim)

            taz_metrics = self._last_taz_metrics_by_id[taz]
            prev_taz_metrics = (
                self._prev_action_taz_metrics.get(taz, self._zero_taz_metrics())
                if self._prev_action_taz_metrics is not None else
                self._zero_taz_metrics()
            )
            t_occ_norm = taz_metrics["mean_occupancy"] / 100.0 if taz_metrics["mean_occupancy"] > 1.5 else taz_metrics["mean_occupancy"]
            t_occ_norm = self._clip01(t_occ_norm)
            waiting_delta_norm = float(np.clip(
                (
                    self._safe_float(taz_metrics.get("total_waiting_time", 0.0), 0.0)
                    - self._safe_float(prev_taz_metrics.get("total_waiting_time", 0.0), 0.0)
                ) / self.taz_waiting_time_ref,
                -5.0,
                5.0,
            ))
            active_delta_norm = float(np.clip(
                (
                    self._safe_float(taz_metrics.get("active_vehicle_count", 0.0), 0.0)
                    - self._safe_float(prev_taz_metrics.get("active_vehicle_count", 0.0), 0.0)
                ) / self.taz_veh_total_ref,
                -5.0,
                5.0,
            ))
            jam_delta_norm = float(np.clip(
                (
                    self._safe_float(taz_metrics.get("max_jam_len", 0.0), 0.0)
                    - self._safe_float(prev_taz_metrics.get("max_jam_len", 0.0), 0.0)
                ) / self.max_jam_len_ref,
                -2.0,
                2.0,
            ))
            speed_delta_norm = float(np.clip(
                (
                    self._safe_float(taz_metrics.get("mean_speed", 0.0), 0.0)
                    - self._safe_float(prev_taz_metrics.get("mean_speed", 0.0), 0.0)
                ) / self.speed_norm,
                -2.0,
                2.0,
            ))
            row.extend(
                [
                    float(np.clip(self._safe_float(taz_metrics.get("max_jam_len", 0.0), 0.0) / self.jam_norm, 0.0, 2.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("mean_speed", 0.0), 0.0) / self.speed_norm, 0.0, 2.0)),
                    float(t_occ_norm),
                    float(np.clip(self._safe_float(taz_metrics.get("veh_total", 0.0), 0.0) / self.taz_veh_total_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("active_vehicle_count", 0.0), 0.0) / self.taz_veh_total_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("total_waiting_time", 0.0), 0.0) / self.taz_waiting_time_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("total_co2", 0.0), 0.0) / self.taz_co2_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("total_nox", 0.0), 0.0) / self.taz_nox_ref, 0.0, 5.0)),
                    hour_sin,
                    hour_cos,
                    waiting_delta_norm,
                    active_delta_norm,
                    jam_delta_norm,
                    speed_delta_norm,
                ]
            )
            self._episode_max_jam_len = max(
                self._episode_max_jam_len,
                self._safe_float(taz_metrics.get("max_jam_len", 0.0), 0.0),
            )
            obs_rows.append(row)

        obs_tensor = torch.tensor(obs_rows, dtype=torch.float32, device=self.device)
        if not torch.isfinite(obs_tensor).all():
            obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
        self._current_action_mask = self._build_runtime_action_mask()
        self._prev_action_taz_metrics = self._clone_taz_metrics_snapshot()
        return obs_tensor

    def _get_output_dir(self) -> Optional[str]:
        root = getattr(self.sumo, "typePath", None)
        if root is None:
            root = getattr(self.sumo, "routeFilePath", None)
        if root is None:
            return None
        return os.path.join(str(root), self.output_subdir)

    def _parse_tripinfo_metrics(self, tripinfo_path: str) -> dict:
        if not os.path.exists(tripinfo_path):
            return {
                "trip_count": 0,
                "avg_waiting_time": 0.0,
                "total_waiting_time": 0.0,
                "ok": False,
            }

        root = self._safe_parse_xml_root(tripinfo_path)
        if root is None:
            return {
                "trip_count": 0,
                "avg_waiting_time": 0.0,
                "total_waiting_time": 0.0,
                "ok": False,
            }

        waiting_times = []
        for trip in root.findall("tripinfo"):
            waiting_times.append(self._safe_float(trip.get("waitingTime"), 0.0))

        if not waiting_times:
            return {
                "trip_count": 0,
                "avg_waiting_time": 0.0,
                "total_waiting_time": 0.0,
                "ok": False,
            }

        total_waiting_time = float(sum(waiting_times))
        return {
            "trip_count": int(len(waiting_times)),
            "avg_waiting_time": float(total_waiting_time / len(waiting_times)),
            "total_waiting_time": total_waiting_time,
            "ok": True,
        }

    def _parse_emission_metric(self, emission_path: str, metric: str) -> dict:
        if not os.path.exists(emission_path):
            return {"total_emission": 0.0, "ok": False}

        root = self._safe_parse_xml_root(emission_path)
        if root is None:
            return {"total_emission": 0.0, "ok": False}

        totals = []
        for timestep in root.findall("timestep"):
            vehicles = timestep.findall("vehicle")
            if vehicles:
                totals.append(
                    sum(
                        self._safe_float(self._get_attr_case_insensitive(vehicle, metric), 0.0)
                        for vehicle in vehicles
                    )
                )
                continue

            total_attr = self._get_attr_case_insensitive(timestep, metric)
            totals.append(self._safe_float(total_attr, 0.0))

        if not totals:
            return {"total_emission": 0.0, "ok": False}

        return {
            "total_emission": float(sum(totals)),
            "ok": True,
        }

    def _normalize_metric(self, value: float, ref: float) -> float:
        return float(np.clip(float(value) / float(ref), 0.0, self.metric_clip))

    def _describe_step_reward_basis(self) -> str:
        dense_terms = []
        if self.dense_abs_penalty_weight > 0.0:
            dense_terms.append("abs")
        if self.dense_delta_reward_weight > 0.0:
            dense_terms.append("delta")
        if not dense_terms:
            dense_terms.append("none")

        metrics = []
        if self.waiting_reward_weight > 0.0:
            metrics.append("waiting")
        if self.emission_reward_weight > 0.0:
            metrics.append("emission")
        if self.jam_reward_weight > 0.0:
            metrics.append("jam")
        metric_label = "+".join(metrics) if metrics else "none"

        basis = f"taz_step_{'+'.join(dense_terms)}_{metric_label}"
        if self.terminal_bonus_weight > 0.0:
            basis += "_with_terminal"
        if self.comparison_reward_enabled:
            basis += "_baseline_compare"
        return basis

    def _compute_terminal_reward(self) -> tuple[float, dict]:
        output_dir = self._get_output_dir()
        if output_dir is None:
            return 0.0, {
                "parse_ok": False,
                "reason": "missing_output_dir",
            }

        tripinfo_path = os.path.join(output_dir, self.tripinfo_filename)
        emission_path = os.path.join(output_dir, self.emission_filename)
        waiting = self._get_episode_waiting_metrics()
        co2 = self._parse_emission_metric(emission_path, "CO2")
        nox = self._parse_emission_metric(emission_path, "NOx")
        waiting_ok = bool(waiting["vehicle_count"] > 0)
        parse_ok = bool(waiting_ok and co2["ok"] and nox["ok"])
        if not parse_ok:
            return 0.0, {
                "parse_ok": False,
                "reason": "missing_or_invalid_waiting_or_emission",
                "tripinfo_path": tripinfo_path,
                "emission_path": emission_path,
            }

        waiting_penalty = self._normalize_metric(waiting["total_waiting_time"], self.terminal_waiting_time_ref)
        co2_penalty = self._normalize_metric(co2["total_emission"], self.terminal_co2_ref)
        nox_penalty = self._normalize_metric(nox["total_emission"], self.terminal_nox_ref)
        emission_penalty = (
            self.emission_co2_mix_weight * co2_penalty
            + self.emission_nox_mix_weight * nox_penalty
        )
        jam_penalty = self._normalize_metric(self._episode_max_jam_len, self.max_jam_len_ref)
        penalty = (
            self.waiting_reward_weight * waiting_penalty
            + self.emission_reward_weight * emission_penalty
            + self.jam_reward_weight * jam_penalty
        )

        use_comparison = bool(self.comparison_reward_enabled and (self.reference_penalty is not None))
        baseline_penalty = float(self.reference_penalty) if use_comparison else 0.0
        reward_raw = baseline_penalty - penalty if use_comparison else -penalty
        reward = float(np.clip(self.terminal_bonus_weight * reward_raw, -self.reward_clip, self.reward_clip))

        components = {
            "parse_ok": True,
            "reward_basis": "reference_waiting_emission_jam_delta" if use_comparison else "absolute_waiting_emission_jam_penalty",
            "trip_count": int(waiting["vehicle_count"]),
            "avg_waiting_time": float(waiting["avg_waiting_time"]),
            "total_waiting_time": float(waiting["total_waiting_time"]),
            "waiting_metric_source": "live_accumulated_waiting_time",
            "waiting_time_by_taz": dict(waiting["waiting_time_by_taz"]),
            "vehicle_count_by_taz": dict(waiting["vehicle_count_by_taz"]),
            "total_co2": float(co2["total_emission"]),
            "total_nox": float(nox["total_emission"]),
            "episode_max_jam_len": float(self._episode_max_jam_len),
            "waiting_penalty": float(waiting_penalty),
            "co2_penalty": float(co2_penalty),
            "nox_penalty": float(nox_penalty),
            "emission_penalty": float(emission_penalty),
            "jam_penalty": float(jam_penalty),
            "penalty": float(penalty),
            "baseline_penalty": float(baseline_penalty),
            "reference_penalty": float(baseline_penalty),
            "delta_penalty": float(baseline_penalty - penalty) if use_comparison else 0.0,
            "reward_raw": float(reward_raw),
            "total": float(reward),
            "tripinfo_path": tripinfo_path,
            "emission_path": emission_path,
        }
        return reward, components

    def run_reference_episode_no_agent(self, hour: int) -> tuple[float, dict]:
        self.set_baseline_penalty(None)
        self.set_episode_context(hour)
        self.reset()

        while True:
            self.sumo.step(quantity=self.stepSize)
            self.current_step += 1
            self._collect_observation()
            sim_running = self.sumo.isRunning()
            time_limit_reached = self.current_step >= self.cooldownSteps
            terminated = (not sim_running) and (not time_limit_reached)
            truncated = bool(time_limit_reached)
            if terminated or truncated:
                break

        if self.sumo.isLoaded():
            self.sumo.end()

        _, baseline_components = self._compute_terminal_reward()
        self.last_reward_components = dict(baseline_components)
        self.last_duration_diagnostics = self.get_episode_duration_diagnostics()
        baseline_penalty = float(baseline_components.get("penalty", 0.0))
        return baseline_penalty, baseline_components

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed

    def _reset(self, tensordict=None, **kwargs):
        if self.sumo.isLoaded():
            self.sumo.end()

        self.sumo.start(
            activeGui=False,
            logFilePath=self.sumo.logFile,
            rl_mode=True,
            waitingTimeMemory=self.waiting_time_memory,
            seed=self.sumo_seed,
            threadRngs=self.sumo_thread_rngs,
        )
        self.current_step = 0
        self._last_green_tls_snapshot = {}
        self._phase_duration_memory = {}
        self._active_program_id = {}
        self._build_tls_lane_cache()
        for tls in self.tls_list:
            self._sync_phase_duration_memory(tls)
        self._reset_duration_diagnostics()
        self.last_reward_components = {}
        self.last_duration_diagnostics = {}
        self._episode_max_jam_len = 0.0
        self._prev_action_taz_metrics = None
        self._action_signal_by_taz = {
            taz: 0.0
            for taz in self.taz_ids
        }
        self._current_action_mask = torch.zeros_like(self._action_mask, dtype=torch.bool, device=self.device)
        self._episode_vehicle_waiting_by_id = {}
        self._episode_vehicle_taz_by_id = {}
        self._last_tls_metrics_by_id = {
            tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
            for tls in self.tls_list
        }
        self._last_taz_metrics_by_id = {
            taz: self._zero_taz_metrics()
            for taz in self.taz_ids
        }

        obs = self._collect_observation()
        penalties, _ = self._build_taz_penalties()
        self._prev_taz_penalties = penalties.copy()
        return TensorDict(
            {
                "observation": obs,
                "action_mask": self.get_action_mask(),
            },
            batch_size=[],
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict["action"]
        if action.requires_grad:
            action = action.detach()
        action = action.to(self.device).reshape(len(self.taz_ids), self.max_tls_per_taz)
        action_index = torch.round(action).long().clamp(0, len(self.ACTION_BIN_VALUES) - 1)
        zero_index = torch.full_like(action_index, self.ZERO_ACTION_INDEX)
        runtime_action_mask = self.get_action_mask()
        action_index = torch.where(runtime_action_mask, action_index, zero_index)
        selected_action = self.decode_action_indices(action_index)

        applied_action = torch.zeros_like(selected_action)
        applied_duration_delta = torch.zeros_like(selected_action)
        for taz_idx, taz in enumerate(self.taz_ids):
            for slot_idx, tls in enumerate(self.tls_by_taz[taz]):
                action_value = float(selected_action[taz_idx, slot_idx].item())
                applied_action[taz_idx, slot_idx] = action_value
                applied_duration_delta[taz_idx, slot_idx] = float(self._apply_action_to_tls(tls, action_value))

        self.sumo.step(quantity=self.stepSize)
        self.current_step += 1
        sim_running = self.sumo.isRunning()
        time_limit_reached = self.current_step >= self.cooldownSteps

        obs = self._collect_observation()
        penalties, penalty_details = self._build_taz_penalties()

        if self.current_step <= self.warmupSteps:
            dense_reward_vector = np.zeros_like(penalties, dtype=np.float32)
        else:
            dense_reward_vector = self._compute_dense_reward_vector(penalties)

        terminated = (not sim_running) and (not time_limit_reached)
        truncated = bool(time_limit_reached)
        terminal_reward = 0.0
        terminal_components = {}
        reward_vector = dense_reward_vector.copy()
        if terminated or truncated:
            if self.sumo.isLoaded():
                self.sumo.end()
            terminal_reward, terminal_components = self._compute_terminal_reward()
            reward_vector = reward_vector + float(terminal_reward)
            self.last_duration_diagnostics = self.get_episode_duration_diagnostics()
        else:
            self.last_duration_diagnostics = {}

        penalty_by_taz = {
            taz: float(penalties[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        dense_reward_by_taz = {
            taz: float(dense_reward_vector[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        reward_by_taz = {
            taz: float(reward_vector[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        delta_penalty_by_taz = {
            taz: float(self._prev_taz_penalties[idx] - penalties[idx])
            for idx, taz in enumerate(self.taz_ids)
        }

        reward_basis = self._describe_step_reward_basis()

        self.last_reward_components = {
            "reward_basis": reward_basis,
            "reward_by_taz": reward_by_taz,
            "dense_reward_by_taz": dense_reward_by_taz,
            "penalty_by_taz": penalty_by_taz,
            "penalty_details_by_taz": penalty_details,
            "delta_penalty_by_taz": delta_penalty_by_taz,
            "reward_mean": float(np.mean(reward_vector)),
            "reward_std": float(np.std(reward_vector)),
            "reward_min": float(np.min(reward_vector)),
            "reward_max": float(np.max(reward_vector)),
            "reward_sum": float(np.sum(reward_vector)),
            "dense_reward_mean": float(np.mean(dense_reward_vector)),
            "dense_reward_sum": float(np.sum(dense_reward_vector)),
            "penalty_mean": float(np.mean(penalties)),
            "penalty_std": float(np.std(penalties)),
            "penalty_min": float(np.min(penalties)),
            "penalty_max": float(np.max(penalties)),
            "penalty_sum": float(np.sum(penalties)),
            "terminal_bonus": float(terminal_reward),
            "terminal_reward": terminal_components,
            "reference_penalty": float(self.reference_penalty) if self.reference_penalty is not None else None,
            "reference_reward_components": dict(self.reference_reward_components),
            "action_signal_by_taz": dict(self._action_signal_by_taz),
            "action_enabled_ratio": float(self._current_action_mask.float().mean().item()),
            "episode_steps": int(self.current_step),
            "num_taz": int(len(self.taz_ids)),
            "num_tls": int(len(self.tls_list)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "warmup": bool(self.current_step <= self.warmupSteps),
        }
        self._prev_taz_penalties = penalties.copy()

        return TensorDict(
            {
                "observation": obs,
                "action_mask": self.get_action_mask(),
                "reward": torch.tensor(reward_vector, dtype=torch.float32, device=self.device),
                "applied_action": applied_action,
                "applied_duration_delta": applied_duration_delta,
                "terminated": torch.tensor(terminated, device=self.device),
                "truncated": torch.tensor(truncated, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )


class GroupedTazTrafficEnv(LocalTazTrafficEnv):
    """
    Grouped TAZ environment:
      - keeps the local TAZ per-TLS observation backbone and reward design
      - replaces per-TLS actions with coordinated nearby-TLS groups
      - applies one shared delta per group using wider {-10, 0, +10} bins
      - within a group, applies the delta only to TLS currently aligned on the
        same dominant green axis to favor corridor-style progression
    """

    ACTION_BIN_VALUES = (-10.0, 0.0, 10.0)
    ZERO_ACTION_INDEX = 1

    def __init__(
        self,
        *args,
        coordination_distance: float = 350.0,
        coordination_path_length: float = 900.0,
        coordination_group_max_size: int = 3,
        group_axis_align_only: bool = True,
        **kwargs,
    ):
        # grouped still inherits the full per-TLS/per-TAZ state from local TAZ, but augments it
        # with topology-aware control groups that become the new action units.
        self.coordination_distance = max(float(coordination_distance), 1.0)
        self.coordination_path_length = max(float(coordination_path_length), 1.0)
        self.coordination_group_max_size = max(int(coordination_group_max_size), 1)
        self.group_axis_align_only = bool(group_axis_align_only)

        super().__init__(*args, **kwargs)

        self.base_agent_obs_dim = int(self.agent_obs_dim)
        self.per_group_feature_dim = 9
        # The network topology is parsed once so each TAZ can be partitioned into stable
        # nearby groups of TLS that can be controlled together.
        self._tls_nodes = self._build_tls_nodes_from_net(SUMO_NET_FILE_PATH)
        self._tls_positions = self._build_tls_positions_from_net(SUMO_NET_FILE_PATH)
        self._road_graph = self._build_road_graph_from_net(SUMO_NET_FILE_PATH)
        self._tls_topology_distance_by_pair = self._build_tls_topology_distances()
        self._tls_link_axis_by_index = self._build_tls_link_axis_by_index(SUMO_NET_FILE_PATH)
        self.control_groups_by_taz = self._build_control_groups_by_taz()
        self.max_control_groups_per_taz = max(len(groups) for groups in self.control_groups_by_taz.values())

        self.agent_obs_dim = (
            self.max_tls_per_taz * self.per_tls_feature_dim
            + self.per_taz_feature_dim
            + self.max_control_groups_per_taz * self.per_group_feature_dim
        )
        # Observation layout in grouped:
        #   [all per-TLS local TAZ features for one TAZ]
        #   + [local TAZ aggregated per-TAZ features]
        #   + [one feature block for each coordinated control group in that same TAZ]
        self.observation_spec = UnboundedSpec(
            shape=(len(self.taz_ids), self.agent_obs_dim),
            device=self.device,
        )
        self.action_spec = BoundedSpec(
            low=0,
            high=len(self.ACTION_BIN_VALUES) - 1,
            shape=(len(self.taz_ids), self.max_control_groups_per_taz),
            device=self.device,
        )

        action_mask = []
        for taz in self.taz_ids:
            valid = len(self.control_groups_by_taz[taz])
            row = [True] * valid + [False] * (self.max_control_groups_per_taz - valid)
            action_mask.append(row)
        # Each row of the mask still corresponds to one TAZ, but actions now index groups
        # instead of raw TLS slots.
        self._group_action_mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        self._current_action_mask = self._group_action_mask.clone()
        self.action_bins = torch.tensor(self.ACTION_BIN_VALUES, dtype=torch.float32, device=self.device)

        self._prev_action_tls_metrics = None
        self._last_tls_axis_by_id = {
            tls: "mixed"
            for tls in self.tls_list
        }
        self._group_signal_by_taz = {
            taz: [0.0 for _ in groups]
            for taz, groups in self.control_groups_by_taz.items()
        }
        self.last_group_action_details = {}

    @staticmethod
    def _axis_from_delta(dx: float, dy: float) -> str:
        return "EW" if abs(dx) >= abs(dy) else "NS"

    def _build_tls_nodes_from_net(self, net_path: str) -> dict[str, set[str]]:
        """Map every TLS to the road-network nodes that best represent its location."""
        tree = ET.parse(net_path)
        root = tree.getroot()

        junction_ids = {
            junction.get("id")
            for junction in root.findall(".//junction")
            if junction.get("id")
        }
        edge_to_to_node = {}
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            to_node = edge.get("to")
            if edge_id and to_node:
                edge_to_to_node[edge_id] = to_node

        tls_nodes = defaultdict(set)
        for connection in root.findall(".//connection"):
            tls_id = connection.get("tl")
            edge_id = connection.get("from")
            if not tls_id:
                continue
            if tls_id in junction_ids:
                tls_nodes[tls_id].add(tls_id)
            if edge_id in edge_to_to_node:
                tls_nodes[tls_id].add(edge_to_to_node[edge_id])

        for tls_id in self.tls_list:
            if tls_id in junction_ids:
                tls_nodes[tls_id].add(tls_id)
            tls_nodes.setdefault(tls_id, set())

        return {
            tls_id: set(node_ids)
            for tls_id, node_ids in tls_nodes.items()
        }

    def _build_tls_positions_from_net(self, net_path: str) -> dict[str, tuple[float, float]]:
        """Estimate one representative 2D position for each TLS from the SUMO net."""
        tree = ET.parse(net_path)
        root = tree.getroot()

        junction_positions = {}
        for junction in root.findall(".//junction"):
            jid = junction.get("id")
            x = junction.get("x")
            y = junction.get("y")
            if jid and x and y:
                junction_positions[jid] = (float(x), float(y))

        edge_to_to_node = {}
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            to_node = edge.get("to")
            if edge_id and to_node:
                edge_to_to_node[edge_id] = to_node

        tls_nodes = defaultdict(list)
        for connection in root.findall(".//connection"):
            tls_id = connection.get("tl")
            edge_id = connection.get("from")
            if tls_id and edge_id in edge_to_to_node:
                tls_nodes[tls_id].append(edge_to_to_node[edge_id])

        tls_positions = {}
        for tls_id in self.tls_list:
            points = []
            if tls_id in junction_positions:
                points.append(junction_positions[tls_id])
            for node_id in tls_nodes.get(tls_id, []):
                if node_id in junction_positions:
                    points.append(junction_positions[node_id])
            if not points:
                tls_positions[tls_id] = (0.0, 0.0)
                continue
            tls_positions[tls_id] = (
                float(sum(x for x, _ in points) / len(points)),
                float(sum(y for _, y in points) / len(points)),
            )
        return tls_positions

    def _build_road_graph_from_net(self, net_path: str) -> dict[str, list[tuple[str, float]]]:
        """Build an undirected road graph so group candidates can be compared topologically."""
        tree = ET.parse(net_path)
        root = tree.getroot()

        junction_positions = {}
        for junction in root.findall(".//junction"):
            jid = junction.get("id")
            x = junction.get("x")
            y = junction.get("y")
            if jid and x and y:
                junction_positions[jid] = (float(x), float(y))

        graph = defaultdict(list)
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            from_node = edge.get("from")
            to_node = edge.get("to")
            function = edge.get("function", "")
            if (
                not edge_id
                or not from_node
                or not to_node
                or function == "internal"
                or edge_id.startswith(":")
            ):
                continue

            length = None
            try:
                length_attr = edge.get("length")
                if length_attr is not None:
                    length = float(length_attr)
            except (TypeError, ValueError):
                length = None

            if length is None:
                lane_lengths = []
                for lane in edge.findall("lane"):
                    lane_length = lane.get("length")
                    if lane_length is None:
                        continue
                    try:
                        lane_lengths.append(float(lane_length))
                    except (TypeError, ValueError):
                        continue
                if lane_lengths:
                    length = float(sum(lane_lengths) / len(lane_lengths))

            if length is None:
                p0 = junction_positions.get(from_node)
                p1 = junction_positions.get(to_node)
                if p0 is not None and p1 is not None:
                    length = self._distance(p0, p1)

            if length is None or not math.isfinite(length):
                length = 1.0
            length = max(float(length), 1.0)

            graph[from_node].append((to_node, length))
            graph[to_node].append((from_node, length))

        return {
            node_id: list(neighbors)
            for node_id, neighbors in graph.items()
        }

    def _build_tls_link_axis_by_index(self, net_path: str) -> dict[str, dict[int, str]]:
        """Infer whether each controlled link of a TLS is mainly east-west or north-south."""
        tree = ET.parse(net_path)
        root = tree.getroot()

        junction_positions = {}
        for junction in root.findall(".//junction"):
            jid = junction.get("id")
            x = junction.get("x")
            y = junction.get("y")
            if jid and x and y:
                junction_positions[jid] = (float(x), float(y))

        edge_axis = {}
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            from_node = edge.get("from")
            to_node = edge.get("to")
            if (
                edge_id
                and from_node in junction_positions
                and to_node in junction_positions
            ):
                x0, y0 = junction_positions[from_node]
                x1, y1 = junction_positions[to_node]
                edge_axis[edge_id] = self._axis_from_delta(x1 - x0, y1 - y0)

        link_axis_by_index = defaultdict(dict)
        for connection in root.findall(".//connection"):
            tls_id = connection.get("tl")
            link_index = connection.get("linkIndex")
            if not tls_id or link_index is None:
                continue

            axis = edge_axis.get(connection.get("from", ""))
            if axis is None:
                axis = edge_axis.get(connection.get("to", ""))
            if axis is None:
                continue

            try:
                link_axis_by_index[tls_id][int(link_index)] = axis
            except (TypeError, ValueError):
                continue

        return {
            tls_id: dict(index_map)
            for tls_id, index_map in link_axis_by_index.items()
        }

    @staticmethod
    def _distance(p0: tuple[float, float], p1: tuple[float, float]) -> float:
        return float(math.hypot(p0[0] - p1[0], p0[1] - p1[1]))

    def _multi_source_shortest_paths(self, source_nodes: set[str]) -> dict[str, float]:
        """Run a bounded Dijkstra search from a set of TLS-related nodes."""
        dist = {}
        heap = []
        for node_id in source_nodes:
            if node_id not in self._road_graph:
                continue
            dist[node_id] = 0.0
            heapq.heappush(heap, (0.0, node_id))

        while heap:
            current_dist, node_id = heapq.heappop(heap)
            if current_dist > dist.get(node_id, float("inf")):
                continue
            if current_dist > self.coordination_path_length:
                continue

            for neighbor_id, edge_length in self._road_graph.get(node_id, []):
                next_dist = current_dist + float(edge_length)
                if next_dist >= dist.get(neighbor_id, float("inf")):
                    continue
                if next_dist > self.coordination_path_length:
                    continue
                dist[neighbor_id] = next_dist
                heapq.heappush(heap, (next_dist, neighbor_id))

        return dist

    def _build_tls_topology_distances(self) -> dict[tuple[str, str], float]:
        """Precompute pairwise TLS distances inside each TAZ using the road graph."""
        pair_distances = {}
        for taz in self.taz_ids:
            tls_ids = list(self.tls_by_taz[taz])
            tls_nodes = {
                tls_id: {
                    node_id
                    for node_id in self._tls_nodes.get(tls_id, set())
                    if node_id in self._road_graph
                }
                for tls_id in tls_ids
            }
            for tls_id in tls_ids:
                source_nodes = tls_nodes.get(tls_id, set())
                if not source_nodes:
                    continue
                node_distances = self._multi_source_shortest_paths(source_nodes)
                for other_tls_id in tls_ids:
                    if other_tls_id == tls_id:
                        pair_distances[(tls_id, other_tls_id)] = 0.0
                        continue

                    target_nodes = tls_nodes.get(other_tls_id, set())
                    if not target_nodes:
                        continue

                    best_distance = min(
                        (node_distances.get(node_id, float("inf")) for node_id in target_nodes),
                        default=float("inf"),
                    )
                    if math.isfinite(best_distance):
                        pair_distances[(tls_id, other_tls_id)] = float(best_distance)

        return pair_distances

    def _build_control_groups_by_taz(self) -> dict[str, list[list[str]]]:
        """
        Partition each TAZ into ordered groups of nearby TLS.

        The grouping logic first prefers topological closeness on the road graph and then
        falls back to Euclidean distance, which gives grouped action groups that are spatially
        coherent corridors rather than arbitrary TLS bundles.
        """
        grouped = {}
        for taz in self.taz_ids:
            remaining = set(self.tls_by_taz[taz])
            groups = []
            while remaining:
                anchor = min(
                    remaining,
                    key=lambda tls: (
                        self._tls_positions.get(tls, (0.0, 0.0))[0],
                        self._tls_positions.get(tls, (0.0, 0.0))[1],
                        tls,
                    ),
                )
                remaining.remove(anchor)
                group = [anchor]

                while remaining and len(group) < self.coordination_group_max_size:
                    connected_candidates = []
                    nearby_candidates = []
                    for tls in remaining:
                        tls_pos = self._tls_positions.get(tls, (0.0, 0.0))
                        nearest_member_dist = min(
                            self._distance(tls_pos, self._tls_positions.get(member, (0.0, 0.0)))
                            for member in group
                        )
                        nearest_topology_dist = min(
                            self._tls_topology_distance_by_pair.get((tls, member), float("inf"))
                            for member in group
                        )

                        common_sort = (
                            self._tls_positions.get(tls, (0.0, 0.0))[0],
                            self._tls_positions.get(tls, (0.0, 0.0))[1],
                            tls,
                        )
                        if nearest_topology_dist <= self.coordination_path_length:
                            connected_candidates.append((nearest_topology_dist, nearest_member_dist, *common_sort))
                        elif nearest_member_dist <= self.coordination_distance:
                            nearby_candidates.append((nearest_member_dist, *common_sort))

                    if connected_candidates:
                        _, _, _, _, best_tls = min(connected_candidates)
                    elif nearby_candidates:
                        _, _, _, best_tls = min(nearby_candidates)
                    else:
                        break
                    remaining.remove(best_tls)
                    group.append(best_tls)

                group.sort(
                    key=lambda tls: (
                        self._tls_positions.get(tls, (0.0, 0.0))[0],
                        self._tls_positions.get(tls, (0.0, 0.0))[1],
                        tls,
                    )
                )
                groups.append(group)

            groups.sort(
                key=lambda group: (
                    self._tls_positions.get(group[0], (0.0, 0.0))[0],
                    self._tls_positions.get(group[0], (0.0, 0.0))[1],
                    group[0],
                )
            )
            grouped[taz] = groups
        return grouped

    def get_control_groups_by_taz(self) -> dict[str, list[list[str]]]:
        return {
            taz: [list(group) for group in groups]
            for taz, groups in self.control_groups_by_taz.items()
        }

    def _clone_tls_metrics_snapshot(self) -> dict[str, dict]:
        """Freeze the latest TLS metrics so group signals can compare consecutive steps."""
        return {
            tls: {
                key: self._safe_float(value, 0.0)
                for key, value in metrics.items()
            }
            for tls, metrics in self._last_tls_metrics_by_id.items()
        }

    def _infer_tls_current_green_axis(self, tls_id: str) -> str:
        """Infer the dominant green direction currently served by one TLS."""
        fallback_axis = self._last_tls_axis_by_id.get(tls_id, "mixed")
        link_axis = self._tls_link_axis_by_index.get(tls_id, {})
        if not link_axis:
            return fallback_axis

        try:
            program, _ = self._get_active_program_logic(tls_id)
        except Exception:
            return fallback_axis
        if program is None:
            return fallback_axis

        phases = list(program.phases)
        if not phases:
            return fallback_axis

        current_phase_id = int(getattr(program, "currentPhaseIndex", 0))
        if current_phase_id < 0 or current_phase_id >= len(phases):
            return fallback_axis

        state = str(getattr(phases[current_phase_id], "state", "") or "")
        ew_green = 0
        ns_green = 0
        for link_idx, phase_char in enumerate(state):
            if phase_char not in {"G", "g"}:
                continue
            axis = link_axis.get(link_idx)
            if axis == "EW":
                ew_green += 1
            elif axis == "NS":
                ns_green += 1

        if ew_green > ns_green:
            return "EW"
        if ns_green > ew_green:
            return "NS"
        return fallback_axis

    def _compute_group_signal(self, group_tls_ids: list[str]) -> float:
        """
        Summarize how much a control group changed since the previous observation.

        This signal is later used both as an observation feature and as a runtime gate
        to disable actions on groups that are currently too inactive to matter.
        """
        if self._prev_action_tls_metrics is None:
            return 0.0

        signals = []
        for tls_id in group_tls_ids:
            current = self._last_tls_metrics_by_id.get(tls_id, self._zero_tls_metrics())
            previous = self._prev_action_tls_metrics.get(tls_id, self._zero_tls_metrics())

            waiting_delta = abs(
                self._safe_float(current.get("total_waiting_time", 0.0), 0.0)
                - self._safe_float(previous.get("total_waiting_time", 0.0), 0.0)
            ) / self.tls_waiting_time_ref
            active_delta = abs(
                self._safe_float(current.get("active_vehicle_count", 0.0), 0.0)
                - self._safe_float(previous.get("active_vehicle_count", 0.0), 0.0)
            ) / self.tls_active_vehicle_ref
            pressure_delta = abs(
                self._safe_float(current.get("pressure", 0.0), 0.0)
                - self._safe_float(previous.get("pressure", 0.0), 0.0)
            ) / self.tls_pressure_ref
            signals.append(waiting_delta + 0.5 * active_delta + 0.25 * pressure_delta)

        if not signals:
            return 0.0
        return float(np.clip(float(sum(signals) / len(signals)), 0.0, 5.0))

    def _build_runtime_action_mask(self) -> torch.Tensor:
        """Disable group actions whose current signal is below the configured threshold."""
        runtime_mask = self._group_action_mask.clone()
        self._group_signal_by_taz = {}
        self._action_signal_by_taz = {}

        for taz_idx, taz in enumerate(self.taz_ids):
            signals = []
            for group_idx, group_tls_ids in enumerate(self.control_groups_by_taz[taz]):
                signal = self._compute_group_signal(group_tls_ids)
                signals.append(signal)
                if self.action_signal_threshold > 0.0 and signal < self.action_signal_threshold:
                    runtime_mask[taz_idx, group_idx] = False
            self._group_signal_by_taz[taz] = list(signals)
            self._action_signal_by_taz[taz] = float(max(signals) if signals else 0.0)

        return runtime_mask

    def _build_group_feature_block(self, taz: str) -> list[float]:
        """
        Build the grouped-specific group feature block for one TAZ.

        Each group contributes 9 values:
          presence, normalized size, mean phase duration, mean active vehicles,
          mean waiting time, mean pressure, EW ratio, NS ratio, and dynamic signal.
        """
        block = []
        groups = self.control_groups_by_taz[taz]
        for group_idx in range(self.max_control_groups_per_taz):
            if group_idx >= len(groups):
                block.extend([0.0] * self.per_group_feature_dim)
                continue

            group_tls_ids = groups[group_idx]
            tls_metrics = [
                self._last_tls_metrics_by_id.get(tls_id, self._zero_tls_metrics(taz_id=taz))
                for tls_id in group_tls_ids
            ]
            group_size = max(len(group_tls_ids), 1)
            ew_count = 0
            ns_count = 0
            for tls_id in group_tls_ids:
                axis = self._last_tls_axis_by_id.get(tls_id, "mixed")
                if axis == "EW":
                    ew_count += 1
                elif axis == "NS":
                    ns_count += 1

            mean_phase_duration = float(np.mean([
                np.clip(self._safe_float(metrics.get("phase_duration", 0.0), 0.0) / float(self.max_green), 0.0, 2.0)
                for metrics in tls_metrics
            ]))
            mean_active = float(np.mean([
                np.clip(self._safe_float(metrics.get("active_vehicle_count", 0.0), 0.0) / self.tls_active_vehicle_ref, 0.0, 5.0)
                for metrics in tls_metrics
            ]))
            mean_waiting = float(np.mean([
                np.clip(self._safe_float(metrics.get("total_waiting_time", 0.0), 0.0) / self.tls_waiting_time_ref, 0.0, 5.0)
                for metrics in tls_metrics
            ]))
            mean_pressure = float(np.mean([
                np.clip(max(self._safe_float(metrics.get("pressure", 0.0), 0.0), 0.0) / self.tls_pressure_ref, 0.0, 5.0)
                for metrics in tls_metrics
            ]))
            signal = float(self._group_signal_by_taz.get(taz, [0.0] * len(groups))[group_idx])

            block.extend(
                [
                    1.0,
                    float(len(group_tls_ids) / float(self.coordination_group_max_size)),
                    mean_phase_duration,
                    mean_active,
                    mean_waiting,
                    mean_pressure,
                    float(ew_count / group_size),
                    float(ns_count / group_size),
                    signal,
                ]
            )
        return block

    def _collect_observation(self) -> torch.Tensor:
        """
        Return one observation row per TAZ.

        The row is strictly local to that TAZ:
          1. the inherited local TAZ backbone with per-TLS and per-TAZ traffic features
          2. the grouped block with coordinated-group features for that same TAZ

        No explicit cross-TAZ global context is concatenated here.
        """
        base_obs = super()._collect_observation()

        if base_obs.ndim != 2 or tuple(base_obs.shape) != (len(self.taz_ids), self.base_agent_obs_dim):
            base_obs = torch.zeros(
                (len(self.taz_ids), self.base_agent_obs_dim),
                dtype=torch.float32,
                device=self.device,
            )

        if not self.sumo.isRunning():
            self._last_tls_axis_by_id = {
                tls: "mixed"
                for tls in self.tls_list
            }
            self._group_signal_by_taz = {
                taz: [0.0 for _ in groups]
                for taz, groups in self.control_groups_by_taz.items()
            }
            self._prev_action_tls_metrics = self._clone_tls_metrics_snapshot()
            group_tensor = torch.zeros(
                (len(self.taz_ids), self.max_control_groups_per_taz * self.per_group_feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
            return torch.cat([base_obs, group_tensor], dim=-1)

        # The current green-axis estimate lets one group action affect only the TLS that
        # are aligned with the dominant direction served at this moment.
        self._last_tls_axis_by_id = {
            tls_id: self._infer_tls_current_green_axis(tls_id)
            for tls_id in self.tls_list
        }

        # One extra feature block is appended for each group slot in the current TAZ.
        group_rows = []
        for taz in self.taz_ids:
            group_rows.append(self._build_group_feature_block(taz))
        group_tensor = torch.tensor(group_rows, dtype=torch.float32, device=self.device)

        if not torch.isfinite(group_tensor).all():
            group_tensor = torch.nan_to_num(group_tensor, nan=0.0, posinf=10.0, neginf=-10.0)

        self._prev_action_tls_metrics = self._clone_tls_metrics_snapshot()
        return torch.cat([base_obs, group_tensor], dim=-1)

    def _select_group_members_for_action(self, group_tls_ids: list[str]) -> tuple[list[str], str]:
        """
        Restrict a group action to the TLS aligned with the dominant served axis.

        When axis alignment is enabled, this keeps one group action from stretching
        semaphores that are currently serving the orthogonal direction.
        """
        if len(group_tls_ids) <= 1 or not self.group_axis_align_only:
            return list(group_tls_ids), "mixed"

        ew_weight = 0.0
        ns_weight = 0.0
        for tls_id in group_tls_ids:
            axis = self._last_tls_axis_by_id.get(tls_id, "mixed")
            weight = 1.0 + self._safe_float(
                self._last_tls_metrics_by_id.get(tls_id, {}).get("total_waiting_time", 0.0),
                0.0,
            ) / self.tls_waiting_time_ref
            if axis == "EW":
                ew_weight += weight
            elif axis == "NS":
                ns_weight += weight

        if ew_weight <= 0.0 and ns_weight <= 0.0:
            return list(group_tls_ids), "mixed"

        dominant_axis = "EW" if ew_weight >= ns_weight else "NS"
        aligned_members = [
            tls_id
            for tls_id in group_tls_ids
            if self._last_tls_axis_by_id.get(tls_id, "mixed") == dominant_axis
        ]
        if not aligned_members:
            return list(group_tls_ids), dominant_axis
        return aligned_members, dominant_axis

    def _step(self, tensordict):
        """
        Apply one coordinated action per group, advance SUMO, and emit the next local state.

        Even though actions are coordinated within each TAZ, the environment still returns
        one reward value and one observation row per TAZ, preserving the local-agent view.
        """
        action = tensordict["action"]
        if action.requires_grad:
            action = action.detach()
        action = action.to(self.device).reshape(len(self.taz_ids), self.max_control_groups_per_taz)
        action_index = torch.round(action).long().clamp(0, len(self.ACTION_BIN_VALUES) - 1)
        zero_index = torch.full_like(action_index, self.ZERO_ACTION_INDEX)
        runtime_action_mask = self.get_action_mask()
        action_index = torch.where(runtime_action_mask, action_index, zero_index)
        selected_action = self.decode_action_indices(action_index)

        applied_action = torch.zeros_like(selected_action)
        applied_duration_delta = torch.zeros_like(selected_action)
        self.last_group_action_details = {}

        # Iterate TAZ by TAZ: each TAZ owns several group-action slots, and each slot can
        # affect one or more TLS depending on the currently dominant movement axis.
        for taz_idx, taz in enumerate(self.taz_ids):
            per_taz_details = []
            for group_idx, group_tls_ids in enumerate(self.control_groups_by_taz[taz]):
                action_value = float(selected_action[taz_idx, group_idx].item())
                applied_action[taz_idx, group_idx] = action_value

                if abs(action_value) <= 1e-9:
                    per_taz_details.append(
                        {
                            "group_index": int(group_idx),
                            "tls_ids": list(group_tls_ids),
                            "selected_tls_ids": [],
                            "dominant_axis": "mixed",
                            "requested_delta": float(action_value),
                            "applied_delta_mean": 0.0,
                        }
                    )
                    continue

                selected_tls_ids, dominant_axis = self._select_group_members_for_action(group_tls_ids)
                member_deltas = [
                    float(self._apply_action_to_tls(tls_id, action_value))
                    for tls_id in selected_tls_ids
                ]
                applied_duration_delta[taz_idx, group_idx] = float(
                    sum(member_deltas) / max(len(member_deltas), 1)
                )
                per_taz_details.append(
                    {
                        "group_index": int(group_idx),
                        "tls_ids": list(group_tls_ids),
                        "selected_tls_ids": list(selected_tls_ids),
                        "dominant_axis": dominant_axis,
                        "requested_delta": float(action_value),
                        "applied_delta_mean": float(applied_duration_delta[taz_idx, group_idx].item()),
                    }
                )
            self.last_group_action_details[taz] = per_taz_details

        # After actions are applied, the next observation is collected with the updated
        # traffic metrics and group features, then the dense and terminal rewards are built.
        self.sumo.step(quantity=self.stepSize)
        self.current_step += 1
        sim_running = self.sumo.isRunning()
        time_limit_reached = self.current_step >= self.cooldownSteps

        obs = self._collect_observation()
        penalties, penalty_details = self._build_taz_penalties()

        if self.current_step <= self.warmupSteps:
            dense_reward_vector = np.zeros_like(penalties, dtype=np.float32)
        else:
            dense_reward_vector = self._compute_dense_reward_vector(penalties)

        terminated = (not sim_running) and (not time_limit_reached)
        truncated = bool(time_limit_reached)
        terminal_reward = 0.0
        terminal_components = {}
        reward_vector = dense_reward_vector.copy()
        if terminated or truncated:
            if self.sumo.isLoaded():
                self.sumo.end()
            terminal_reward, terminal_components = self._compute_terminal_reward()
            reward_vector = reward_vector + float(terminal_reward)
            self.last_duration_diagnostics = self.get_episode_duration_diagnostics()
        else:
            self.last_duration_diagnostics = {}

        penalty_by_taz = {
            taz: float(penalties[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        dense_reward_by_taz = {
            taz: float(dense_reward_vector[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        reward_by_taz = {
            taz: float(reward_vector[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        delta_penalty_by_taz = {
            taz: float(self._prev_taz_penalties[idx] - penalties[idx])
            for idx, taz in enumerate(self.taz_ids)
        }

        reward_basis = self._describe_step_reward_basis()

        self.last_reward_components = {
            "reward_basis": reward_basis,
            "reward_by_taz": reward_by_taz,
            "dense_reward_by_taz": dense_reward_by_taz,
            "penalty_by_taz": penalty_by_taz,
            "penalty_details_by_taz": penalty_details,
            "delta_penalty_by_taz": delta_penalty_by_taz,
            "reward_mean": float(np.mean(reward_vector)),
            "reward_std": float(np.std(reward_vector)),
            "reward_min": float(np.min(reward_vector)),
            "reward_max": float(np.max(reward_vector)),
            "reward_sum": float(np.sum(reward_vector)),
            "dense_reward_mean": float(np.mean(dense_reward_vector)),
            "dense_reward_sum": float(np.sum(dense_reward_vector)),
            "penalty_mean": float(np.mean(penalties)),
            "penalty_std": float(np.std(penalties)),
            "penalty_min": float(np.min(penalties)),
            "penalty_max": float(np.max(penalties)),
            "penalty_sum": float(np.sum(penalties)),
            "terminal_bonus": float(terminal_reward),
            "terminal_reward": terminal_components,
            "reference_penalty": float(self.reference_penalty) if self.reference_penalty is not None else None,
            "reference_reward_components": dict(self.reference_reward_components),
            "action_signal_by_taz": dict(self._action_signal_by_taz),
            "group_signal_by_taz": {
                taz: list(signals)
                for taz, signals in self._group_signal_by_taz.items()
            },
            "group_action_details": dict(self.last_group_action_details),
            "action_enabled_ratio": float(self._current_action_mask.float().mean().item()),
            "episode_steps": int(self.current_step),
            "num_taz": int(len(self.taz_ids)),
            "num_tls": int(len(self.tls_list)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "warmup": bool(self.current_step <= self.warmupSteps),
        }
        self._prev_taz_penalties = penalties.copy()

        return TensorDict(
            {
                "observation": obs,
                "action_mask": self.get_action_mask(),
                "reward": torch.tensor(reward_vector, dtype=torch.float32, device=self.device),
                "applied_action": applied_action,
                "applied_duration_delta": applied_duration_delta,
                "terminated": torch.tensor(terminated, device=self.device),
                "truncated": torch.tensor(truncated, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )


class CoordinatedTazTrafficEnv(GroupedTazTrafficEnv):
    """Local TAZ controller plus online TAZ-to-TAZ flow tracking.

    The grouped base environment already manages observations, rewards, and
    signal-duration actions. This subclass only adds a small memory of each
    active vehicle's last observed TAZ, so the coordinator can receive sparse
    TAZ-to-TAZ flow diagnostics after every SUMO step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._taz_index = {taz_id: idx for idx, taz_id in enumerate(self.taz_ids)}
        self._coord_vehicle_last_taz: dict[str, str] = {}
        self._coord_last_step_flow = np.zeros((len(self.taz_ids), len(self.taz_ids)), dtype=np.float32)
        self._coord_episode_flow = np.zeros_like(self._coord_last_step_flow)

    def _reset_coordination_trackers(self):
        """Clear vehicle memory and sparse flow matrices at episode start."""

        self._coord_vehicle_last_taz = {}
        self._coord_last_step_flow = np.zeros((len(self.taz_ids), len(self.taz_ids)), dtype=np.float32)
        self._coord_episode_flow = np.zeros_like(self._coord_last_step_flow)

    def _reset(self, tensordict=None, **kwargs):
        self._reset_coordination_trackers()
        return super()._reset(tensordict=tensordict, **kwargs)

    def _record_vehicle_taz_transition(self, vehicle_id: str, current_taz: str | None):
        """Count a transition when a vehicle is observed in a different TAZ."""

        if current_taz is None or current_taz not in self._taz_index:
            return
        vehicle_id = str(vehicle_id)
        previous_taz = self._coord_vehicle_last_taz.get(vehicle_id)
        self._coord_vehicle_last_taz[vehicle_id] = str(current_taz)
        if previous_taz is None or previous_taz == current_taz or previous_taz not in self._taz_index:
            return
        src_idx = self._taz_index[previous_taz]
        dst_idx = self._taz_index[str(current_taz)]
        self._coord_last_step_flow[src_idx, dst_idx] += 1.0
        self._coord_episode_flow[src_idx, dst_idx] += 1.0

    def _collect_vehicle_metrics_by_tls(self) -> dict:
        """Mirror the parent metrics collector and add TAZ transition tracking."""

        self._coord_last_step_flow = np.zeros((len(self.taz_ids), len(self.taz_ids)), dtype=np.float32)
        metrics_by_tls = {
            tls: {
                "active_vehicle_count": 0.0,
                "total_waiting_time": 0.0,
                "avg_waiting_time": 0.0,
                "total_co2": 0.0,
                "total_nox": 0.0,
            }
            for tls in self.tls_list
        }
        if not self.sumo.isRunning():
            return metrics_by_tls

        try:
            vehicle_ids = list(libtraci.vehicle.getIDList())
        except Exception:
            return metrics_by_tls

        active_vehicle_ids = set()
        for vehicle_id in vehicle_ids:
            vehicle_id = str(vehicle_id)
            active_vehicle_ids.add(vehicle_id)
            try:
                lane_id = str(libtraci.vehicle.getLaneID(vehicle_id))
            except Exception:
                lane_id = ""
            tls_idx = self._lane_to_tls_idx.get(lane_id)
            if tls_idx is None:
                tls_idx = self._edge_to_tls_idx.get(self._lane_to_edge_id(lane_id))
            if tls_idx is None:
                try:
                    road_id = str(libtraci.vehicle.getRoadID(vehicle_id))
                except Exception:
                    road_id = ""
                tls_idx = self._edge_to_tls_idx.get(road_id)
            if tls_idx is None:
                continue

            try:
                waiting_time = self._safe_float(libtraci.vehicle.getAccumulatedWaitingTime(vehicle_id), 0.0)
                co2 = self._safe_float(libtraci.vehicle.getCO2Emission(vehicle_id), 0.0)
                nox = self._safe_float(libtraci.vehicle.getNOxEmission(vehicle_id), 0.0)
            except Exception:
                continue

            tls_id = self.tls_list[int(tls_idx)]
            taz_id = self.tls_to_taz.get(tls_id)
            self._record_vehicle_taz_transition(vehicle_id, str(taz_id) if taz_id is not None else None)
            prev_waiting = self._safe_float(self._episode_vehicle_waiting_by_id.get(vehicle_id, 0.0), 0.0)
            self._episode_vehicle_waiting_by_id[vehicle_id] = float(max(prev_waiting, waiting_time))
            if taz_id is not None:
                self._episode_vehicle_taz_by_id[vehicle_id] = str(taz_id)
            metrics_by_tls[tls_id]["active_vehicle_count"] += 1.0
            metrics_by_tls[tls_id]["total_waiting_time"] += float(waiting_time)
            metrics_by_tls[tls_id]["total_co2"] += float(co2)
            metrics_by_tls[tls_id]["total_nox"] += float(nox)

        stale_vehicle_ids = set(self._coord_vehicle_last_taz) - active_vehicle_ids
        for vehicle_id in stale_vehicle_ids:
            self._coord_vehicle_last_taz.pop(vehicle_id, None)
        for tls_id in self.tls_list:
            count = metrics_by_tls[tls_id]["active_vehicle_count"]
            if count > 0:
                metrics_by_tls[tls_id]["avg_waiting_time"] = metrics_by_tls[tls_id]["total_waiting_time"] / float(count)
        return metrics_by_tls

    def get_last_step_flow_matrix(self) -> np.ndarray:
        """Return latest observed TAZ-to-TAZ flow matrix."""

        return self._coord_last_step_flow.copy()

    def get_episode_flow_matrix(self) -> np.ndarray:
        """Return cumulative TAZ-to-TAZ flow matrix for the current episode."""

        return self._coord_episode_flow.copy()

    def get_coordination_flow_diagnostics(self) -> dict:
        """Expose sparse flow dictionaries for JSON reports."""

        return {
            "last_step_flow_by_taz": self._matrix_to_nested_dict(self._coord_last_step_flow),
            "episode_flow_by_taz": self._matrix_to_nested_dict(self._coord_episode_flow),
        }

    def _matrix_to_nested_dict(self, matrix: np.ndarray) -> dict:
        return {
            src_taz: {
                dst_taz: float(matrix[src_idx, dst_idx])
                for dst_taz, dst_idx in self._taz_index.items()
                if float(matrix[src_idx, dst_idx]) > 0.0
            }
            for src_taz, src_idx in self._taz_index.items()
        }

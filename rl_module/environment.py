from __future__ import annotations

"""Named coordinated-control environment used by the RL package."""

import libtraci
import numpy as np

from taz_rl.rlenv.local_taz_env_v11 import SumoTazEnvV11


class CoordinatedTazTrafficEnv(SumoTazEnvV11):
    """Local TAZ controller plus online TAZ-to-TAZ flow tracking."""

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


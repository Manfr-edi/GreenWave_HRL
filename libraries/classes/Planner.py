from __future__ import annotations

"""Minimal route-planning API required by the RL training scripts.

The original project planner also interacts with databases and digital-twin
components. The standalone RL module only needs route generation, so this file
keeps the public API used by `rl_module` while avoiding unrelated dependencies.
"""

import os
import subprocess
import sys
from sys import platform

from libraries import constants
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import SUMO_NET_FILE_PATH, SUMO_ROUTES_PATH, SUMO_TOOLS_PATH, SUMO_TOOLS_PATH_UBUNTU


def _resolve_sumo_tool(tool_name: str) -> str:
    """Find a SUMO Python tool without depending on the original project path."""

    candidate_roots = []
    if os.environ.get("SUMO_TOOLS"):
        candidate_roots.append(os.environ["SUMO_TOOLS"])
    if os.environ.get("SUMO_HOME"):
        candidate_roots.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    if platform == "linux" or platform == "linux2":
        candidate_roots.append(SUMO_TOOLS_PATH_UBUNTU)
    candidate_roots.append(SUMO_TOOLS_PATH)

    for root in candidate_roots:
        if not root:
            continue
        tool_path = os.path.join(root, tool_name)
        if os.path.exists(tool_path):
            return tool_path
    searched = ", ".join(candidate_roots)
    raise FileNotFoundError(f"Cannot find SUMO tool '{tool_name}'. Searched: {searched}")


class ScenarioGenerator:
    """Generate SUMO route files from edge-count observations."""

    def __init__(self, sumocfg: str, sim: Simulator):
        self.sumoConfiguration = sumocfg
        self.sim = sim

    def setScenario(self, routeFilePath=None, manual: bool = False, absolutePath: bool = False):
        """Point the simulator to an existing route folder or route file."""

        if manual:
            raise NotImplementedError("Manual file selection is intentionally not included in the standalone package.")
        if not routeFilePath:
            raise ValueError("No route file path was provided.")
        if not absolutePath:
            routeFilePath = os.path.abspath(routeFilePath)
        if not os.path.exists(routeFilePath):
            raise FileNotFoundError(f"The route path '{routeFilePath}' does not exist.")
        self.sim.changeRoutePath(routePath=routeFilePath)

    def generateRandomRoute(
        self,
        sumoNetPath: str,
        timeSlot: str,
        custom: bool = True,
        outputFolder: str | None = None,
        seed: int | None = None,
    ):
        """Create random candidate trips that `routeSampler.py` can later fit to edge counts."""

        del timeSlot
        folder_path = os.path.abspath(outputFolder) if outputFolder else os.path.join(SUMO_ROUTES_PATH, "default")
        os.makedirs(folder_path, exist_ok=True)

        command = [
            sys.executable,
            _resolve_sumo_tool("randomTrips.py"),
            "-n",
            sumoNetPath,
            "-r",
            os.path.join(folder_path, "randomTrips.rou.xml"),
            "--output-trip-file",
            os.path.join(folder_path, "trips.rou.xml"),
            "--random-departpos",
            "--random-arrivalpos",
            "--allow-fringe",
            "--remove-loops",
            "--fringe-factor",
            "10",
            "--min-distance",
            "100",
            "--max-distance",
            "2000",
            "--random-routing-factor",
            "10",
            "--period",
            "0.1",
        ]
        if custom:
            command.extend(["--trip-attributes", "type='customModel'"])
        if seed is None:
            command.append("--random")
        else:
            command.extend(["--seed", str(int(seed))])
        subprocess.run(command, check=True)

    def generateRoute(
        self,
        inputEdgePath: str,
        timeSlot: str,
        withInitialRoute=True,
        totalCount=10000,
        custom: bool = True,
        outputFolder: str | None = None,
        randomTripSeed: int | None = None,
        routeSamplerSeed: int | None = None,
        routeSamplerThreads: int = 8,
    ):
        """Fit generated candidate trips to the observed edge counts for one episode."""

        time_slot_folder = timeSlot.replace(":", "-")
        folder_path = os.path.abspath(outputFolder) if outputFolder else os.path.join(constants.SUMO_ROUTES_PATH, time_slot_folder)
        os.makedirs(folder_path, exist_ok=True)

        if withInitialRoute:
            self.generateRandomRoute(
                sumoNetPath=SUMO_NET_FILE_PATH,
                timeSlot=time_slot_folder,
                custom=custom,
                outputFolder=folder_path,
                seed=randomTripSeed,
            )

        command = [
            sys.executable,
            _resolve_sumo_tool("routeSampler.py"),
            "--r",
            os.path.join(folder_path, "randomTrips.rou.xml"),
            "--edgedata-files",
            inputEdgePath,
            "-o",
            os.path.join(folder_path, "generatedRoutes.rou.xml"),
            "--edgedata-attribute",
            "entered",
            "--write-flows",
            "number",
            "--total-count",
            str(totalCount),
            "--optimize",
            "full",
            "--minimize-vehicles",
            "1",
            "--threads",
            str(int(routeSamplerThreads)),
            "--verbose",
        ]
        if custom:
            command.extend(["--attributes", "type='customModel'"])
        if routeSamplerSeed is not None:
            command.extend(["--seed", str(int(routeSamplerSeed))])
        subprocess.run(command, check=True)


class Planner:
    """Small facade kept for compatibility with the RL scripts."""

    def __init__(self, simulator: Simulator):
        self.simulator = simulator
        self.scenarioGenerator = ScenarioGenerator(sumocfg="run.sumocfg", sim=self.simulator)

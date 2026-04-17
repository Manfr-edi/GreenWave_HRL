# ****************************************************
# Module Purpose:
#   This library defines the Simulator class, which is responsible for interacting with the sumoenv Simulator.
#   The Simulator class manages the running of the simulation, from its start to its completion.
#
#   It provides various methods for controlling the simulation, gathering data on vehicles, induction loops,
#   and managing traffic lights using the libtraci library for communication with the sumoenv Simulator and the
#   running simulation inside it.
#
# ****************************************************

from statistics import mean
import os
# These libraries are quite the same. They include most of the commands available in the traci library, but the
# time performance better. Libtraci addresses some limitation of the libsumo library, in particular the multiple client
# communication with the simulation.
import libtraci
import traci.constants as tc
from typing import Optional
import pandas as pd
import json

from libraries.constants import SUMO_OUTPUT_PATH, SUMO_PATH, SUMO_NETWORK_PATH


class Simulator:
    """
    The Simulator class provides an interface to interact with the sumoenv traffic simulation
    environment using the libtraci library.

    Attributes:
        configurationPath (str): Path to the sumoenv configuration file.
        routePath (str): Path to the route file used in the simulation.
        logFile (str): Path to the file where logs are saved.
        listener (libtraci.StepListener): A listener object for simulation steps.
        vehicleSummary (dict): A dictionary to store vehicle data summaries.

    Class Methods:
        - __init__: Constructor to initialize a new instance of the Simulator class.
        - start: Method to start the sumoenv simulation with or without the GUI.
        - isRunning: Method to check if the simulation is running.
        - startBasic: Method to start the basic sumoenv simulation configuration.
        - startCongestioned: Method to start the sumoenv simulation with congestion.
        - step: Method to execute a defined number of simulation steps.
        - oneHourStep: Method to advance the simulation by one hour (3600 seconds).
        - resume: Method to resume the simulation until no more vehicles remain.
        - end: Method to end the simulation and close the connection to sumoenv.
        - getRemainingVehicles: Method to retrieve the number of remaining vehicles in the simulation.
        - changeRoutePath: Method to change the route path for the simulation.
        - getVehiclesSummary: Method to gather and return vehicle statistics from the simulation.
        - getDetectorList: Method to get a list of all induction loop detectors.
        - getAverageOccupationTime: Method to calculate the average occupation time for all detectors.
        - getInductionLoopSummary: Method to calculate summary statistics for induction loops.
        - findLinkedTLS: Method to find traffic light systems (TLS) linked to a detector.
        - subscribeToInductionLoop: Method to subscribe to induction loop data.
        - checkSubscription: Method to monitor subscription results for induction loops and modify traffic light programs.
        - getTLSList: Method to retrieve a list of all traffic light systems (TLS).
        - checkTLS: Method to check if a given TLS exists in the simulation.
        - setTLSProgram: Method to set the traffic light program for a TLS or all TLSs.
    """

    logFile: str
    def __init__(self, configurationPath: str, logFile: str, tazTlsMapFile = None):
        """
        Initializes the Simulator with the given configuration file and log file.

        :param configurationPath: Path to the sumoenv configuration file.
        :param logFile: Path to the log file where simulation logs will be saved.
        """
        self.configurationPath = configurationPath
        self.logFile = logFile
        self.tazTlsMapFile = tazTlsMapFile
        # TODO: check if this routePath variable is needed.
        self.routePath = configurationPath
        self.typePath = configurationPath

        if tazTlsMapFile is not None:
            with open(tazTlsMapFile) as f:
                self.tazTlsMap = json.load(f)

        # WRONG PATH
        #staticpath = os.path.abspath(self.configurationPath + "/static")
        if not os.path.exists(SUMO_NETWORK_PATH):
            print("Error: the given path does not exist.")
            return
        outputpath = os.path.abspath(self.configurationPath + "/output")
        os.makedirs(outputpath, exist_ok=True)

        os.environ["STATICPATH"] = SUMO_NETWORK_PATH
        self.listener = ValueListener()
        # Registering many listeners across recreated Simulator instances may leak resources.
        # Keep the listener object available but do not auto-register it here.

    def start(self, activeGui: bool = False, logFilePath: Optional[str] = None, noWarnings: bool = True,
              continuous: bool = False, rl_mode: bool = False, traceCommands: bool = False,
              waitingTimeMemory: Optional[int] = None, seed: Optional[int] = None,
              threadRngs: Optional[int] = None):
        """
        Start the SUMO environment simulation, with or without the GUI, based on the `activeGui` parameter.
        If a simulation is already loaded, it will be overwritten.

        :param activeGui: If True, starts the simulation with the SUMO GUI (sumo-gui).
                          If False, starts the simulation without the GUI (sumo). Default is False.
        :param logFilePath: Optional path to a log file. If specified, the log file is used for the SUMO trace.
        :raises RuntimeError: If there is an issue starting the SUMO simulation.
        """
        # Always close any previously loaded simulation before starting a new one.
        if libtraci.simulation.isLoaded():
            print("Warning: A previous simulation was loaded. Closing it before restart.")
            self.end()

        # Construct the command for starting SUMO or SUMO-GUI
        sumo_command = "sumo-gui" if activeGui else "sumo"
        if noWarnings:
            if rl_mode:
                command = [sumo_command, "-c", os.path.join(self.configurationPath, "run_rl.sumocfg"), "--no-step-log",
                           "true", "-W", "true", "--duration-log.disable"]
            else:
                command = [sumo_command, "-c", os.path.join(self.configurationPath, "run.sumocfg"), "--no-step-log",
                       "true", "-W", "true", "--duration-log.disable"]

        else:
            if rl_mode:
                command = [sumo_command, "-c", os.path.join(self.configurationPath, "run_rl.sumocfg")]
            else:
                command = [sumo_command, "-c", os.path.join(self.configurationPath, "run.sumocfg")]
        if waitingTimeMemory is not None:
            command.extend(["--waiting-time-memory", str(max(int(waitingTimeMemory), 1))])
        if seed is not None:
            command.extend(["--seed", str(int(seed))])
        if threadRngs is not None:
            command.extend(["--thread-rngs", str(max(int(threadRngs), 1))])
        # Set the log file path if specified
        self.logFile = logFilePath if logFilePath else self.logFile

        # Start the simulation. TraCI command tracing can become huge; keep it optional.
        if traceCommands:
            libtraci.start(command, traceFile=self.logFile)
        else:
            libtraci.start(command)
        print("Note: Each simulation step is equivalent to " + str(libtraci.simulation.getDeltaT()) + " seconds.")

        if continuous:
            # Resume the simulation
            self.resume()

    def isRunning(self) -> bool:
        """
        Method to check if the simulation is running. Returns `True` if the simulation is running, `False` otherwise.
        """
        return True if libtraci.simulation.isLoaded() and libtraci.simulation.getMinExpectedNumber() else False

    def isLoaded(self) -> bool:
        """
        Returns True when a TraCI simulation is currently loaded.
        """
        return libtraci.simulation.isLoaded()

    def startBasic(self, activeGui=False):
        """
        Starts the sumoenv simulation with a basic configuration.
        If a simulation is already loaded, it will be overwritten.

        :param activeGui: If True, starts the simulation with the sumoenv GUI (sumo-gui).
                         If False, starts the simulation without the GUI (sumo). Default is False.
        :raises RuntimeError: If there is an issue starting the sumoenv simulation.
        """

        # TODO: CHECK PERCHE' SEMBRA NON FUNZIONARE
        if libtraci.simulation.isLoaded():
            print("Warning: there was a previous simulation loaded. It will be overwritten")
        command = ["sumo-gui" if activeGui else "sumo", "-c", self.configurationPath + "/basic/run.sumocfg", "--log", self.logFile]
        libtraci.start(command, traceFile=self.logFile)
        self.resume()

    def startCongestioned(self, activeGui=False):
        """
        Starts the sumoenv simulation with a congestioned scenario.
        If a simulation is already loaded, it will be overwritten.

        :param activeGui: If True, starts the simulation with the sumoenv GUI (sumo-gui).
                         If False, starts the simulation without the GUI (sumo). Default is False.
        :raises RuntimeError: If there is an issue starting the sumoenv simulation.
        """

        if libtraci.simulation.isLoaded():
            print("Warning: there was a previous simulation loaded. It will be overwritten")
        command = ["sumo-gui" if activeGui else "sumo", "-c", self.configurationPath + "/congestioned/run.sumocfg"]
        libtraci.start(command, traceFile=self.logFile)
        self.resume()

    def step(self, quantity=1):
        """
        Executes a defined number of simulation steps.
        Typically, one step corresponds to one second of simulation time.

        For large `quantity`, a batched TraCI step is attempted first to reduce
        Python-call overhead. If that fails, it falls back to iterative stepping.

        :param quantity: Number of simulation steps to execute.
        """
        qty = int(quantity)
        if qty <= 0:
            return
        if self.getRemainingVehicles() <= 0:
            return

        # Fast path: advance directly to target simulation time in a single call.
        # This is much faster than 100s of Python->TraCI calls per env step.
        if qty > 1:
            try:
                current_time = float(libtraci.simulation.getTime())
                delta_t = float(libtraci.simulation.getDeltaT())
                target_time = current_time + qty * delta_t
                libtraci.simulationStep(target_time)
                return
            except Exception:
                pass

        # Fallback path: iterative stepping.
        step = 0
        while step < qty and self.getRemainingVehicles() > 0:
            libtraci.simulationStep()
            step += 1

    def oneHourStep(self):
        """
        Executes a one-hour simulation step. This advances the simulation by 3600 seconds if there are vehicles
        still in the simulation.

        :raises RuntimeError: If there is an issue performing the one-hour step.
        """
        if libtraci.simulation.getMinExpectedNumber() > 0:
            libtraci.simulationStep(3600)

    def resume(self):
        """
        Resumes the simulation, running continuously until no more vehicles remain in the simulation.
        After the simulation completes, the connection to sumoenv is closed.

        :raises RuntimeError: If the simulation fails to resume or end.
        """
        while libtraci.simulation.getMinExpectedNumber() > 0:
            self.step()
        self.end()

    def end(self):
        """
        Ends the simulation and closes the connection to sumoenv.

        :return: True if the connection was successfully closed, False otherwise.
        :raises RuntimeError: If there is an issue closing the connection.
        """
        if not libtraci.simulation.isLoaded():
            return False
        try:
            libtraci.close()
            return True
        except Exception as e:
            print(f"[WARN] Error while closing SUMO/TraCI session: {e}")
            return False

    def getRemainingVehicles(self):
        """
        Returns the number of vehicles remaining in the simulation, including vehicles waiting to start.

        :return (int): The number of remaining vehicles.
        """
        return libtraci.simulation.getMinExpectedNumber()

    def changeRoutePath(self, routePath: str):
        """
        Changes the route path for the simulator.

        This function checks if the provided route path is absolute. If it is not an absolute path,
        it converts it to an absolute path based on the current working directory.
        After ensuring that the path exists, it updates the simulator's route path and the
        environment variable 'ROUTEFILENAME' to reflect the new path.

        :param routePath: The absolute route file path.
        :raises FileNotFoundError: If the given route path does not exist

        """
        if not os.path.exists(routePath):
            print("Error: the given path does not exist.")
            return
        self.routePath = routePath
        os.environ["ROUTEFILEPATH"] = routePath
        print("The path was set to " + routePath)

    def changeTypePath(self, typePath: str):
        """
        Changes the route path for the simulator.

        This function checks if the provided route path is absolute. If it is not an absolute path,
        it converts it to an absolute path based on the current working directory.
        After ensuring that the path exists, it updates the simulator's route path and the
        environment variable 'ROUTEFILENAME' to reflect the new path.

        :param typePath: The absolute route file path.
        :raises FileNotFoundError: If the given route path does not exist

        """
        if not os.path.exists(typePath):
            print("Error: the given path does not exist.")
            return
        self.typePath = typePath
        os.environ["TYPEPATH"] = typePath
        print("The path was set to " + typePath)


    def changeRouteFilePath(self, routeFilePath: str):
        """
        Changes the route path for the simulator.

        This function checks if the provided route path is absolute. If it is not an absolute path,
        it converts it to an absolute path based on the current working directory.
        After ensuring that the path exists, it updates the simulator's route path and the
        environment variable 'ROUTEFILENAME' to reflect the new path.
        Args:
            :param routePath: The absolute route file path.
        :raises FileNotFoundError: If the given route path does not exist

        """
        if not os.path.exists(routeFilePath):
            print("Error: the given path does not exist.")
            return
        self.routeFilePath = routeFilePath
        os.environ["ROUTEFILEPATH"] = routeFilePath
        print("The path was set to " + routeFilePath)


    def changeDetectorPath(self, detectorPath: str):
        """
        Changes the detector addional file path for the simulator.

        This function checks if the provided route path is absolute. If it is not an absolute path,
        it converts it to an absolute path based on the current working directory.
        After ensuring that the path exists, it updates the simulator's detector path and the
        environment variable 'DETECTORPATH' to reflect the new path.

        :param routePath: The absolute route file path.
        :raises FileNotFoundError: If the given route path does not exist

        """
        if not os.path.exists(detectorPath):
            print("Error: the given path does not exist.")
            return
        self.detectorPath = detectorPath
        os.environ["DETECTORPATH"] = detectorPath
        print("The path was set to " + detectorPath)

    ### VEHICLE FUNCTIONS
    def getVehiclesSummary(self):
        """
        Retrieves a summary of statistics for all vehicles currently in the simulation. The statistics include
        average speed, time lost, distance traveled, departure delay, and waiting time.

        :return (dict): A dictionary containing average statistics for the vehicles in the simulation.
        :raises RuntimeError: If there are no vehicles in the simulation.
        """

        vehicleSummary = {}
        vehiclesList = libtraci.vehicle.getIDList()
        summary = []
        if len(vehiclesList) > 1:
            for vehicleID in vehiclesList:
                element = {}
                element["speed"] = libtraci.vehicle.getSpeed(vehicleID)
                element["timeLost"] = libtraci.vehicle.getTimeLoss(vehicleID)
                element["distance"] = libtraci.vehicle.getDistance(vehicleID)
                element["departDelay"] = libtraci.vehicle.getDepartDelay(vehicleID)
                element["totalWaitingTime"] = libtraci.vehicle.getAccumulatedWaitingTime(vehicleID)
                summary.append(element)

            vehicleSummary["averageSpeed"] = mean(element["speed"] for element in summary)
            vehicleSummary["averageTimeLost"] = mean(element["timeLost"] for element in summary)
            vehicleSummary["averageDepartDelay"] = mean(element["departDelay"] for element in summary)
            vehicleSummary["averageWaitingTime"] = mean(element["totalWaitingTime"] for element in summary)
            # print("The Average Speed of Vehicles is: " + str(vehicleSummary["averageSpeed"]) + " m/s.")
            # print("The Average Time lost is " + str(vehicleSummary["averageTimeLost"]) + " seconds.")
            # print("The Average depart delay is " + str(vehicleSummary["averageDepartDelay"]) + " seconds.")
            # print("The Average time waited is " + str(vehicleSummary["averageWaitingTime"]) + " seconds.")
            self.vehicleSummary = vehicleSummary
            return vehicleSummary
        print("There are no vehicles ")
        return None

    # def updateSummary(self):
    #     # Not sure if it's useful include this data inside Simulator class
    #     self.vehicleSummary = self.getVehiclesSummary()

    ### INDUCTION LOOP FUNCTIONS
    def getDetectorList(self):
        """
        Returns a list of all induction loop detectors in the simulation.

        :return (list): A list of detector IDs.
        """
        return libtraci.inductionloop.getIDList()

    def getAverageOccupationTime(self):
        """
        Calculates and returns the average occupation time for all induction loop detectors in the simulation.

        :return (float): The average occupation time for the detectors.
        """
        detectorList = self.getDetectorList()
        intervalOccupancies = []
        for detector in detectorList:
            intervalOccupancies.append(libtraci.inductionloop.getIntervalOccupancy(detector))
        average = mean(intervalOccupancies)
        return average

    def getInductionLoopSummary(self):
        """
        Retrieves and calculates a summary of statistics for all induction loops in the simulation, including
        interval occupancy, mean speed, and vehicle numbers.

        :return (dict): A dictionary containing average statistics for the induction loops.
        """
        detectorList = self.getDetectorList()
        detectors = []
        inductionLoopSummary = {}
        for det in detectorList:
            element = {}
            element["intervalOccupancy"] = libtraci.inductionloop.getIntervalOccupancy(det)
            element["meanSpeed"] = libtraci.inductionloop.getIntervalMeanSpeed(det)
            element["vehicleNumber"] = libtraci.inductionloop.getIntervalVehicleNumber(det)
            detectors.append(element)
        inductionLoopSummary["averageIntervalOccupancy"] = mean(element["intervalOccupancy"] for element in detectors)
        inductionLoopSummary["averageMeanSpeed"] = mean(element["meanSpeed"] for element in detectors)
        inductionLoopSummary["averageVehicleNumber"] = mean(element["vehicleNumber"] for element in detectors)
        return inductionLoopSummary

    def findLinkedTLS(self, detectorID: str):
        """
        Finds the traffic light systems (TLS) linked to a given detector by matching the lane controlled by the detector.

        :param detectorID: The ID of the detector.
        :return (list): A list of TLS IDs linked to the detector.
        """
        lane = libtraci.inductionloop.getLaneID(detectorID)
        tls = self.getTLSList()
        found = []
        for element in tls:
            lanes = libtraci.trafficlight.getControlledLanes(element)
            if lane in lanes:
                found.append(element)

        return found

    def subscribeToInductionLoop(self, inductionLoopID, value: str):
        """
        Subscribes to an induction loop to monitor specified parameters (occupancy, speed, vehicle number).

        :param inductionLoopID: The ID of the induction loop.
        :param value: The parameter to subscribe to ('intervalOccupancy', 'meanSpeed', 'vehicleNumber').
        :raises ValueError: If the specified value is not a valid parameter for subscription.
        """
        if value == "intervalOccupancy":
            libtraci.inductionloop.subscribe(inductionLoopID, [libtraci.constants.VAR_INTERVAL_OCCUPANCY])
        elif value == "meanSpeed":
            libtraci.inductionloop.subscribe(inductionLoopID, [libtraci.constants.VAR_INTERVAL_SPEED])
        elif value == "vehicleNumber":
            libtraci.inductionloop.subscribe(inductionLoopID, [libtraci.constants.VAR_INTERVAL_NUMBER])

    def checkSubscription(self):
        """
        Checks the subscription results for all induction loops and modifies traffic light programs if
        vehicle numbers or occupancy exceed specified thresholds.
        """
        results = libtraci.inductionloop.getAllSubscriptionResults()
        for key, value in results.items():
            #checking if vehicle number is high
            if libtraci.constants.VAR_INTERVAL_NUMBER in value and value[libtraci.constants.VAR_INTERVAL_NUMBER] > 10:
                tlsIDs = self.findLinkedTLS(key)
                for element in tlsIDs:
                    self.setTLSProgram(element, "utopia")
                    print("New program is " + str(libtraci.trafficlight.getProgram(element)))
            if libtraci.constants.VAR_INTERVAL_OCCUPANCY in value and value[
                libtraci.constants.VAR_INTERVAL_OCCUPANCY] > 30:
                print("value in excess")

    ### TLS FUNCTIONS
    def getTLSList(self):
        """
        Retrieves a list of all traffic light systems (TLS) in the simulation.

        :return (list): A list of TLS IDs.
        """
        return libtraci.trafficlight_getIDList()

    def checkTLS(self, tlsID):
        """
        Checks if a given traffic light system (TLS) exists in the simulation.

        :param tlsID: The ID of the traffic light system.
        :return (bool): True if the TLS exists, False otherwise.
        """
        tls = self.getTLSList()
        return True if tlsID in tls else False


    def get_tls_for_taz(self, taz_id):
        if self.tazTlsMap is None:
            raise RuntimeError("TAZ→TLS mapping not loaded.")
        if taz_id not in self.tazTlsMap:
            raise KeyError(f"TAZ {taz_id} not found in mapping.")
        return self.tazTlsMap[taz_id]

    def setTLSProgram(self, trafficLightID: str, programID: str, all=False):
        """
        Sets the traffic light program for one or all traffic light systems (TLS).

        :param trafficLightID: The ID of the TLS to change the program for.
        :param programID: The ID of the new traffic light program.
        :param all: If True, sets the program for all TLS in the simulation. Default is False.
        :raises RuntimeError: If there is an issue changing the traffic light program.
        """
        ### NOTE: There is still no check of existence of specific program inside the additional file
        if all:
            tls = self.getTLSList()
            for traffic_light in tls:
                libtraci.trafficlight.setProgram(traffic_light, programID)
            print("The program of all traffic lights is changed to" + str(programID))
        elif self.checkTLS(trafficLightID):
            libtraci.trafficlight.setProgram(trafficLightID, programID)
            print("The program of the TLS " + str(trafficLightID) + " is changed to " + str(programID))

    def set_tls_phase(self, tl_id, phase_index):
        libtraci.trafficlight.setPhase(tl_id, phase_index)

    def set_tls_current_phase_duration(self, tl_id, phase_duration, verbose = False):
        program = libtraci.trafficlight_getAllProgramLogics(tl_id)
        phase_index = program[0].currentPhaseIndex
        phase = program[0].phases[phase_index]


        program[0].phases[phase_index].maxDur = phase_duration
        program[0].phases[phase_index].minDur = phase_duration
        program[0].phases[phase_index].duration = phase_duration
        libtraci.trafficlight.setProgramLogic(tl_id, program[0])
        if verbose:
            print("TL with ID: " + str(tl_id) + " phase: " + str(phase_index) + " duration set to: " + str(phase_duration))


    def set_tls_phase_duration(self, tl_id, phase_id, phase_duration, verbose = False):
        program = libtraci.trafficlight.getAllProgramLogics(tl_id)
        phase_index = program[0].currentPhaseIndex
        phase = program[0].phases[phase_index]

        program[0].phases[phase_id].maxDur = phase_duration
        program[0].phases[phase_id].minDur = phase_duration
        program[0].phases[phase_id].duration = phase_duration
        libtraci.trafficlight_setProgramLogic(tl_id, program[0])
        #libtraci.trafficlight.setProgramLogic(tl_id, program[0])
        program = libtraci.trafficlight.getAllProgramLogics(tl_id)
        if verbose:
            print("TL with ID: " + str(tl_id) + " phase: " + str(phase_id) + " duration set to: " + str(phase_duration))

# ------------ E2 DETECTOR FUNCTIONS -----------------

    def get_e2_detectors(self):
        """
        Returns the list of all E2 detector IDs.
        """
        return libtraci.lanearea.getIDList()

    def get_tls_lanes(self, tls_id):
        """
        Returns the lanes controlled by a traffic light logic.
        """
        controlled_links = libtraci.trafficlight.getControlledLinks(tls_id)
        # controlled_links is list[list[(incoming, outgoing, via)]]
        lanes = set()
        for group in controlled_links:
            for link in group:
                incoming = link[0]
                lanes.add(incoming)
        return lanes

    def get_detectors_for_tls(self, tls_id):
        """
        Returns list of E2 detector IDs that belong to the TLS (matching lanes).
        """
        tls_lanes = self.get_tls_lanes(tls_id)
        e2_list = self.get_e2_detectors()

        detectors = []
        for det in e2_list:
            lane = libtraci.lanearea.getLaneID(det)
            if lane in tls_lanes:
                detectors.append(det)
        return detectors

    # ------------ METRICHE ULTIMO INTERVALLO -----------------

    def get_e2_last_interval_metrics(self, tls_id, as_dataframe=False):
        """
        Returns last-interval metrics for all E2 detectors associated with the TLS.
        Metrics:
        - last interval nVehEntered
        - last interval mean speed
        - last interval occupancy (%)
        """
        dets = self.get_detectors_for_tls(tls_id)

        data = []
        for det in dets:
            n = libtraci.lanearea.getLastIntervalVehicleNumber(det)
            speed = libtraci.lanearea.getLastIntervalMeanSpeed(det)
            max_jam_length = libtraci.lanearea.getLastIntervalMaxJamLengthInMeters(det)
            occupancy = libtraci.lanearea.getLastIntervalOccupancy(det)

            data.append({
                "detector": det,
                "lane": libtraci.lanearea.getLaneID(det),
                "nVeh": n,
                "meanSpeed": speed,
                "maxJamLength": max_jam_length,
                "occupancy": occupancy,
            })

        return pd.DataFrame(data) if as_dataframe else data

    # ------------ METRICHE INTERVALLO CORRENTE -----------------

    def get_e2_current_interval_metrics(self, tls_id, as_dataframe=False):
        """
        Current-interval metrics (non-aggregated):
        - # vehicles since simulation start or last reset
        - mean speed
        - occupancy
        """
        dets = self.get_detectors_for_tls(tls_id)

        data = []
        for det in dets:
            n = libtraci.lanearea.getIntervalVehicleNumber(det)
            speed = libtraci.lanearea.getIntervalMeanSpeed(det)
            max_jam_length = libtraci.lanearea.getIntervalMaxJamLengthInMeters(det)
            occupancy = libtraci.lanearea.getIntervalOccupancy(det)

            data.append({
                "detector": det,
                "lane": libtraci.lanearea.getLaneID(det),
                "nVehCurrent": n,
                "meanSpeedCurrent": speed,
                "maxJamLength": max_jam_length,
                "occupancyCurrent": occupancy,
            })

        return pd.DataFrame(data) if as_dataframe else data

    def get_taz_e2_metrics(self, taz_id, interval="last", mode="dict"):
        """
        interval: "last" or "current"
        mode: "dict" or "df"

        Returns:
        - metrics per TLS
        - one average for the entire TAZ
        """
        tls_list = self.get_tls_for_taz(taz_id)

        tls_results = {}
        all_detector_records = []  # serve per media per TAZ
        flat_records = []  # serve per dataframe finale

        for tls in tls_list:
            # ---- Recupero metriche E2 ----
            if interval == "last":
                det_metrics = self.get_e2_last_interval_metrics(tls, as_dataframe=False)
            else:
                det_metrics = self.get_e2_current_interval_metrics(tls, as_dataframe=False)

            # salva dati grezzi TLS
            tls_results[tls] = det_metrics

            # prepara records per la media TAZ e DF
            for r in det_metrics:
                r2 = {"taz": taz_id, "tls": tls}
                r2.update(r)

                flat_records.append(r2)
                all_detector_records.append(r2)


        # ---- Calcolo media unica per tutta la TAZ ----

        # Filtra record validi (solo quelli con veicoli)
        valid_records = [r for r in all_detector_records if r.get("meanSpeed", 0) > 0]
        if len(valid_records) > 0:

            numeric_fields = [
                k for k in all_detector_records[0].keys()
                if k not in ("taz", "tls", "detector", "lane")
            ]

            taz_avg = {"detector": "_taz_avg", "tls": None, "lane": None, "taz": taz_id}

            for field in numeric_fields:
                taz_avg[field] = sum(r[field] for r in all_detector_records) / len(all_detector_records)
        else:
            # Nessun veicolo in tutta la TAZ → valori neutri
            taz_avg = {
                "taz": taz_id,
                "detector": "_taz_avg",
                "tls": None,
                "lane": None,
                "nVeh": 0,
                "meanSpeed": 0,
                "maxJamLength": 0,
                "occupancy": 0
            }

        # aggiungi la riga media al df
        flat_records.append(taz_avg)

        # ---- Output finale ----
        if mode == "dict":
            return {
                "tls_data": tls_results,
                "taz_avg": taz_avg
            }
        else:
            return pd.DataFrame(flat_records)
class ValueListener(libtraci.StepListener):
    """
    A class for defining actions to be executed at every simulation step via the libtraci step listener.
    """
    def step(self, t=0):
        """
        Method called at every simulation step. It allows custom operations to be performed.

        :param t: The simulation time for the current step. Default is 0.
        :return (bool): True to indicate the listener should remain active for the next step.
        """

        # do something after every call to simulationStep
        # print("ExampleListener called with parameter %s." % t)
        # Here it is possible to get every kind of info required during onestep.

        # indicate that the step listener should stay active in the next step
        return True

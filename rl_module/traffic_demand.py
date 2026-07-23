from __future__ import annotations

"""Demand randomization and route-cache management.

The seed value is stable, but demand is still varied deterministically by mixing
the base seed with episode metadata. This gives reproducible randomness without
colliding with the old deterministic route caches.
"""

import hashlib
import csv
import os
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta

from libraries.constants import EDGE_DATA_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, SUMO_OUTPUT_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile

from rl_module.settings import DemandSettings, ExperimentSettings


WEEKDAY_ALIASES = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}
_TRAFFIC_DATE_CACHE: dict[str, set[str]] = {}
_PROCESSED_DATA_DIR = os.path.dirname(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH)


@dataclass(frozen=True)
class EpisodeDemand:
    """Resolved demand and route-cache metadata for one episode."""

    episode_idx: int
    simulation_date: str
    day_type: str
    hour: int
    timeslot: str
    timeslot_clean: str
    hourly_profile_multiplier: float
    day_type_demand_factor: float
    base_vehicle_count: int
    demand_noise: float
    vehicle_count: int
    random_trip_seed: int
    route_sampler_seed: int
    route_folder_path: str
    traffic_flow_file_path: str = ""


def _stable_int(text: str) -> int:
    # Python's built-in hash is salted per process; SHA keeps paths reproducible.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _normalize_day_token(value: str) -> str:
    """Normalize Italian/English weekday names so accents do not matter."""

    normalized = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _parse_explicit_weekdays(day_filter: str) -> set[int] | None:
    """Parse filters such as 'lun-mer-sab' into Python weekday indices."""

    tokens = [
        _normalize_day_token(token)
        for token in re.split(r"[-,;_/\s]+", str(day_filter).strip().lower())
        if token.strip()
    ]
    if not tokens:
        return None
    weekdays = set()
    for token in tokens:
        if token not in WEEKDAY_ALIASES:
            return None
        weekdays.add(WEEKDAY_ALIASES[token])
    return weekdays


def _matches_day_filter(day: datetime, day_filter: str) -> bool:
    normalized = _normalize_day_token(day_filter)
    if normalized == "all":
        return True
    if normalized == "weekdays":
        return day.weekday() < 5
    if normalized == "weekends":
        return day.weekday() >= 5
    explicit_weekdays = _parse_explicit_weekdays(day_filter)
    if explicit_weekdays is not None:
        return day.weekday() in explicit_weekdays
    raise ValueError(
        "day_filter must be one of: all, weekdays, weekends, or explicit weekdays like 'lun-mer-sab'."
    )


def _normalize_data_date(value: str) -> str | None:
    """Normalize supported CSV date tokens to the ISO format used by SUMO route generation."""

    text = str(value).strip()
    if not text:
        return None
    text = text.split()[0]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return text


def resolve_traffic_flow_file_for_year(year: int) -> str:
    """Return the processed traffic-flow CSV that should be used for a simulation year."""

    year = int(year)
    if year == 2026:
        path = os.path.join(_PROCESSED_DATA_DIR, "processed_traffic_flow_2026.csv")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"2026 traffic-flow CSV not found: {path}")
    return PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH


def resolve_traffic_flow_file_for_day(day: datetime) -> str:
    """Return the processed traffic-flow CSV for the given simulation day."""

    return resolve_traffic_flow_file_for_year(int(day.year))


def _load_available_traffic_dates(input_file: str = PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH) -> set[str]:
    """Read the traffic-flow CSV once and return dates that have at least one row."""

    path = os.path.abspath(input_file)
    cached = _TRAFFIC_DATE_CACHE.get(path)
    if cached is not None:
        return cached
    if not os.path.exists(path):
        raise FileNotFoundError(f"Traffic-flow CSV not found: {path}")

    dates = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if "data" not in (reader.fieldnames or []):
            raise ValueError("Traffic-flow CSV must contain a 'data' column.")
        for row in reader:
            normalized = _normalize_data_date(row.get("data", ""))
            if normalized:
                dates.add(normalized)
    if not dates:
        raise ValueError(f"No usable dates found in traffic-flow CSV: {path}")
    _TRAFFIC_DATE_CACHE[path] = dates
    return dates


def _has_traffic_rows(day: datetime, available_dates: set[str] | None = None) -> bool:
    """Return True only when the requested simulation date exists in the CSV."""

    if available_dates is None:
        available_dates = _load_available_traffic_dates(resolve_traffic_flow_file_for_day(day))
    return day.strftime("%Y-%m-%d") in available_dates


def build_episode_schedule(settings: ExperimentSettings) -> list[tuple[datetime, int]]:
    """Build exactly settings.train_days episodes, skipping dates without traffic rows."""

    target_episodes = int(settings.train_days)
    if target_episodes <= 0:
        raise ValueError("train_days must be > 0 because it is the target number of training episodes.")
    if not settings.focus_hours:
        raise ValueError("focus_hours must contain at least one hour.")

    episodes = []
    if settings.train_on_same_day:
        day = settings.start_date
        max_calendar_scan = 366
        for day_offset in range(max_calendar_scan):
            candidate = settings.start_date + timedelta(days=day_offset)
            if _matches_day_filter(candidate, settings.day_filter) and _has_traffic_rows(candidate):
                day = candidate
                break
        else:
            raise ValueError(
                f"Could not find a valid training day with traffic rows after scanning {max_calendar_scan} days. "
                "Check start_date, day_filter, and the traffic-flow CSV."
            )
        for episode_idx in range(target_episodes):
            episodes.append((day, int(settings.focus_hours[episode_idx % len(settings.focus_hours)])))
    else:
        # Calendar days are scanned until the requested number of episodes is reached.
        # A matching day contributes one episode per focus hour, capped at target_episodes.
        # Missing traffic-flow dates are skipped before route generation, so SUMO
        # never receives a date that cannot produce edge data.
        day_offset = 0
        max_calendar_scan = max(366, target_episodes * 14)
        try:
            first_year_end = settings.start_date.replace(year=settings.start_date.year + 1)
        except ValueError:
            first_year_end = settings.start_date + timedelta(days=365)
        preferred_calendar_scan = max(1, (first_year_end - settings.start_date).days)
        while day_offset < max_calendar_scan:
            day = settings.start_date + timedelta(days=day_offset)
            if _matches_day_filter(day, settings.day_filter) and _has_traffic_rows(day):
                for hour in settings.focus_hours:
                    episodes.append((day, int(hour)))
                    if not settings.shuffle_episode_schedule and len(episodes) >= target_episodes:
                        break
            day_offset += 1
            if not settings.shuffle_episode_schedule and len(episodes) >= target_episodes:
                break
            if (
                settings.shuffle_episode_schedule
                and day_offset >= preferred_calendar_scan
                and len(episodes) >= target_episodes
            ):
                break
        if len(episodes) < target_episodes:
            raise ValueError(
                f"Could only build {len(episodes)} episodes after scanning {max_calendar_scan} calendar days. "
                "Check day_filter, focus_hours, and missing dates in the traffic-flow CSV."
            )

    if not episodes:
        raise ValueError("Episode schedule is empty.")
    if settings.shuffle_episode_schedule:
        # Shuffling samples the requested episode count from the full scanned
        # calendar pool while preserving the requested focus-hour balance, then
        # avoids presenting focus hours in a fixed pattern.
        rng = random.Random(settings.global_seed)
        candidate_days = []
        seen_days = set()
        for day, _hour in episodes:
            day_key = day.strftime("%Y-%m-%d")
            if day_key in seen_days:
                continue
            seen_days.add(day_key)
            candidate_days.append(day)
        rng.shuffle(candidate_days)
        episodes = []
        for day in candidate_days:
            for hour in settings.focus_hours:
                episodes.append((day, int(hour)))
                if len(episodes) >= target_episodes:
                    break
            if len(episodes) >= target_episodes:
                break
        rng.shuffle(episodes)
    return episodes[:target_episodes]


def sample_demand_noise(settings: DemandSettings, global_seed: int, episode_idx: int, simulation_date: str, hour: int) -> float:
    """Sample deterministic demand noise from the base seed plus episode metadata."""

    low, high = settings.demand_noise_range
    if float(low) == float(high):
        return float(low)
    key = f"{global_seed}|{episode_idx}|{simulation_date}|{hour}"
    rng = random.Random(global_seed + _stable_int(key))
    return float(rng.uniform(float(low), float(high)))


def _episode_day_type(day: datetime) -> str:
    """Classify calendar days for demand scaling."""

    return "weekend" if day.weekday() >= 5 else "weekday"


def _profile_for_day(day: datetime, demand: DemandSettings) -> dict[int, float]:
    """Pick the hourly profile used to compute the episode demand baseline."""

    if _episode_day_type(day) == "weekend" and demand.weekend_hourly_profile:
        return demand.weekend_hourly_profile
    return demand.hourly_profile


def _hourly_profile_multiplier(day: datetime, hour: int, demand: DemandSettings) -> float:
    """Resolve the hour multiplier, falling back to the standard profile when needed."""

    profile = _profile_for_day(day, demand)
    if int(hour) in profile:
        return float(profile[int(hour)])
    return float(demand.hourly_profile[int(hour)])


def _day_type_demand_factor(day: datetime, demand: DemandSettings) -> float:
    """Apply lower global demand on Saturday/Sunday without changing route seeds."""

    if _episode_day_type(day) == "weekend":
        return float(demand.weekend_demand_factor)
    return 1.0


def resolve_episode_demand(
    episode_idx: int,
    day: datetime,
    hour: int,
    experiment: ExperimentSettings,
    demand: DemandSettings,
) -> EpisodeDemand:
    """Resolve vehicle count, seeds, and route folder for one episode."""

    simulation_date = day.strftime("%Y-%m-%d")
    timeslot = f"{hour:02d}:00-{(hour + 1):02d}:00"
    timeslot_clean = timeslot.replace(":", "-")
    day_type = _episode_day_type(day)
    hourly_multiplier = _hourly_profile_multiplier(day, int(hour), demand)
    day_type_factor = _day_type_demand_factor(day, demand)
    base_vehicle_count = int(demand.base_demand * hourly_multiplier)
    demand_noise = sample_demand_noise(demand, experiment.global_seed, episode_idx, simulation_date, int(hour))
    vehicle_count = max(1, int(round(base_vehicle_count * demand_noise * day_type_factor)))
    random_trip_seed = int(demand.random_trip_seed)
    route_sampler_seed = int(demand.route_sampler_seed)
    if demand.vary_route_seeds_by_episode:
        # Disabled by default: the user asked to keep route seeds unchanged.
        random_trip_seed = int(demand.random_trip_seed + _stable_int(f"trip|{episode_idx}|{simulation_date}|{hour}") % 100000)
        route_sampler_seed = int(demand.route_sampler_seed + _stable_int(f"sample|{episode_idx}|{simulation_date}|{hour}") % 100000)
    route_folder_path = build_route_cache_folder(
        demand.route_cache_root,
        simulation_date,
        timeslot_clean,
        day_type,
        vehicle_count,
        demand_noise,
        day_type_factor,
        experiment.global_seed,
        random_trip_seed,
        route_sampler_seed,
        demand.route_sampler_threads,
    )
    traffic_flow_file_path = resolve_traffic_flow_file_for_day(day)
    return EpisodeDemand(
        episode_idx=int(episode_idx),
        simulation_date=simulation_date,
        day_type=day_type,
        hour=int(hour),
        timeslot=timeslot,
        timeslot_clean=timeslot_clean,
        hourly_profile_multiplier=float(hourly_multiplier),
        day_type_demand_factor=float(day_type_factor),
        base_vehicle_count=int(base_vehicle_count),
        demand_noise=float(demand_noise),
        vehicle_count=int(vehicle_count),
        random_trip_seed=int(random_trip_seed),
        route_sampler_seed=int(route_sampler_seed),
        route_folder_path=route_folder_path,
        traffic_flow_file_path=traffic_flow_file_path,
    )


def build_route_cache_folder(
    root: str,
    simulation_date: str,
    timeslot_clean: str,
    day_type: str,
    vehicle_count: int,
    demand_noise: float,
    day_type_demand_factor: float,
    global_seed: int,
    random_trip_seed: int,
    route_sampler_seed: int,
    route_sampler_threads: int,
) -> str:
    """Build a compact but unique route-cache path for one demand scenario."""

    noise_tag = f"n{float(demand_noise):.4f}".replace(".", "p")
    day_factor_tag = f"df{float(day_type_demand_factor):.4f}".replace(".", "p")
    scenario_tag = "_".join(
        (
            str(day_type),
            f"d{int(vehicle_count)}",
            noise_tag,
            day_factor_tag,
            f"g{int(global_seed)}",
            f"rt{int(random_trip_seed)}",
            f"rs{int(route_sampler_seed)}",
            f"th{int(route_sampler_threads)}",
        )
    )
    return os.path.join(
        root,
        simulation_date,
        timeslot_clean,
        scenario_tag,
    )


def route_files_ready(route_folder_path: str) -> bool:
    """Return True when the required SUMO route files exist and generated routes are valid XML."""

    required = ("randomTrips.rou.xml", "trips.rou.xml", "generatedRoutes.rou.xml")
    if not all(os.path.exists(os.path.join(route_folder_path, name)) for name in required):
        return False
    try:
        ET.parse(os.path.join(route_folder_path, "generatedRoutes.rou.xml"))
    except ET.ParseError:
        return False
    return True


def ensure_routes(episode: EpisodeDemand, planner, demand: DemandSettings) -> tuple[str, bool]:
    """Generate deterministic-randomized routes unless a matching cache already exists."""

    os.makedirs(os.path.join(episode.route_folder_path, "output"), exist_ok=True)
    if demand.reuse_cached_routes and route_files_ready(episode.route_folder_path):
        return episode.route_folder_path, False
    generated_routes_path = os.path.join(episode.route_folder_path, "generatedRoutes.rou.xml")
    if os.path.exists(generated_routes_path):
        os.remove(generated_routes_path)
    traffic_flow_file_path = episode.traffic_flow_file_path or resolve_traffic_flow_file_for_year(
        int(str(episode.simulation_date).split("-", maxsplit=1)[0])
    )
    generateEdgeDataFile(traffic_flow_file_path, date=episode.simulation_date, time_slot=episode.timeslot)
    planner.scenarioGenerator.generateRoute(
        inputEdgePath=EDGE_DATA_FILE_PATH,
        timeSlot=episode.timeslot_clean,
        totalCount=episode.vehicle_count,
        custom=False,
        outputFolder=episode.route_folder_path,
        randomTripSeed=episode.random_trip_seed,
        routeSamplerSeed=episode.route_sampler_seed,
        routeSamplerThreads=demand.route_sampler_threads,
    )
    ET.parse(generated_routes_path)
    return episode.route_folder_path, True


def cleanup_sumo_output_files(*type_paths: str, include_global: bool = True) -> dict:
    """Delete bulky SUMO output files after metrics have already been parsed.

    Route files are kept intact. For each provided ``type_path`` only the nested
    ``output`` directory is cleaned; this removes files such as tripinfos,
    emissions, vehroute, summary, and edge data without touching cached routes.
    ``include_global`` also removes detector output written under
    ``sumoenv/output`` by additional detector files.
    """

    output_dirs = []
    for type_path in type_paths:
        if not type_path:
            continue
        output_dirs.append(os.path.join(os.path.abspath(type_path), "output"))
    if include_global:
        output_dirs.append(os.path.abspath(SUMO_OUTPUT_PATH))

    seen = set()
    files_removed = 0
    bytes_removed = 0
    errors = []
    for output_dir in output_dirs:
        if output_dir in seen or not os.path.isdir(output_dir):
            continue
        seen.add(output_dir)
        for root, _, filenames in os.walk(output_dir):
            for filename in filenames:
                path = os.path.join(root, filename)
                try:
                    size = os.path.getsize(path)
                    os.remove(path)
                    files_removed += 1
                    bytes_removed += int(size)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
    return {
        "files_removed": int(files_removed),
        "bytes_removed": int(bytes_removed),
        "errors": errors,
    }

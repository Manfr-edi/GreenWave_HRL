from __future__ import annotations

"""Demand randomization and route-cache management.

The seed value is stable, but demand is still varied deterministically by mixing
the base seed with episode metadata. This gives reproducible randomness without
colliding with the old deterministic route caches.
"""

import hashlib
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta

from libraries.constants import EDGE_DATA_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH
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


@dataclass(frozen=True)
class EpisodeDemand:
    """Resolved demand and route-cache metadata for one episode."""

    episode_idx: int
    simulation_date: str
    hour: int
    timeslot: str
    timeslot_clean: str
    base_vehicle_count: int
    demand_noise: float
    vehicle_count: int
    random_trip_seed: int
    route_sampler_seed: int
    route_folder_path: str


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


def build_episode_schedule(settings: ExperimentSettings) -> list[tuple[datetime, int]]:
    """Build exactly settings.train_days episodes, optionally shuffled with the fixed seed."""

    target_episodes = int(settings.train_days)
    if target_episodes <= 0:
        raise ValueError("train_days must be > 0 because it is the target number of training episodes.")
    if not settings.focus_hours:
        raise ValueError("focus_hours must contain at least one hour.")

    episodes = []
    if settings.train_on_same_day:
        if not _matches_day_filter(settings.start_date, settings.day_filter):
            raise ValueError("start_date is incompatible with day_filter")
        for episode_idx in range(target_episodes):
            episodes.append((settings.start_date, int(settings.focus_hours[episode_idx % len(settings.focus_hours)])))
    else:
        # Calendar days are scanned until the requested number of episodes is reached.
        # A matching day contributes one episode per focus hour, capped at target_episodes.
        day_offset = 0
        max_calendar_scan = max(366, target_episodes * 14)
        while len(episodes) < target_episodes and day_offset < max_calendar_scan:
            day = settings.start_date + timedelta(days=day_offset)
            if _matches_day_filter(day, settings.day_filter):
                for hour in settings.focus_hours:
                    episodes.append((day, int(hour)))
                    if len(episodes) >= target_episodes:
                        break
            day_offset += 1
        if len(episodes) < target_episodes:
            raise ValueError(
                f"Could only build {len(episodes)} episodes after scanning {max_calendar_scan} calendar days. "
                "Check day_filter and focus_hours."
            )

    if not episodes:
        raise ValueError("Episode schedule is empty.")
    if settings.shuffle_episode_schedule:
        # Shuffling avoids presenting all 07:00/08:00 episodes in a fixed pattern.
        rng = random.Random(settings.global_seed)
        rng.shuffle(episodes)
    return episodes


def sample_demand_noise(settings: DemandSettings, global_seed: int, episode_idx: int, simulation_date: str, hour: int) -> float:
    """Sample deterministic demand noise from the base seed plus episode metadata."""

    low, high = settings.demand_noise_range
    if float(low) == float(high):
        return float(low)
    key = f"{global_seed}|{episode_idx}|{simulation_date}|{hour}"
    rng = random.Random(global_seed + _stable_int(key))
    return float(rng.uniform(float(low), float(high)))


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
    base_vehicle_count = int(demand.base_demand * demand.hourly_profile[int(hour)])
    demand_noise = sample_demand_noise(demand, experiment.global_seed, episode_idx, simulation_date, int(hour))
    vehicle_count = max(1, int(round(base_vehicle_count * demand_noise)))
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
        vehicle_count,
        demand_noise,
        experiment.global_seed,
        random_trip_seed,
        route_sampler_seed,
        demand.route_sampler_threads,
    )
    return EpisodeDemand(
        episode_idx=int(episode_idx),
        simulation_date=simulation_date,
        hour=int(hour),
        timeslot=timeslot,
        timeslot_clean=timeslot_clean,
        base_vehicle_count=int(base_vehicle_count),
        demand_noise=float(demand_noise),
        vehicle_count=int(vehicle_count),
        random_trip_seed=int(random_trip_seed),
        route_sampler_seed=int(route_sampler_seed),
        route_folder_path=route_folder_path,
    )


def build_route_cache_folder(
    root: str,
    simulation_date: str,
    timeslot_clean: str,
    vehicle_count: int,
    demand_noise: float,
    global_seed: int,
    random_trip_seed: int,
    route_sampler_seed: int,
    route_sampler_threads: int,
) -> str:
    """Use explicit path components so route caches document the randomization source."""

    noise_tag = f"noise_{float(demand_noise):.4f}".replace(".", "p")
    return os.path.join(
        root,
        simulation_date,
        timeslot_clean,
        f"demand_{int(vehicle_count)}",
        noise_tag,
        f"base_seed_{int(global_seed)}",
        f"tripseed_{int(random_trip_seed)}_sampleseed_{int(route_sampler_seed)}_threads_{int(route_sampler_threads)}",
    )


def route_files_ready(route_folder_path: str) -> bool:
    """Return True when the required SUMO route files exist."""

    return all(
        os.path.exists(os.path.join(route_folder_path, name))
        for name in ("randomTrips.rou.xml", "trips.rou.xml", "generatedRoutes.rou.xml")
    )


def ensure_routes(episode: EpisodeDemand, planner, demand: DemandSettings) -> tuple[str, bool]:
    """Generate deterministic-randomized routes unless a matching cache already exists."""

    os.makedirs(os.path.join(episode.route_folder_path, "output"), exist_ok=True)
    if demand.reuse_cached_routes and route_files_ready(episode.route_folder_path):
        return episode.route_folder_path, False
    generateEdgeDataFile(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, date=episode.simulation_date, time_slot=episode.timeslot)
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
    return episode.route_folder_path, True

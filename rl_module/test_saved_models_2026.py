"""Run the requested 2026 random-date evaluations for saved RL trainings."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

COMPARE_SCRIPT = os.path.join(PACKAGE_DIR, "compare_coordinated_controller.py")
SAVED_TRAININGS_DIR = os.path.join(PACKAGE_DIR, "saved_trainings")

MODEL_SPECS = {
    # "current_weekday_on_peak": ("__current__", "weekdays", (8, 17)),
    "weekday_on_peak": ("200_weekdays_8_17", "weekdays", (8, 17)),
    # "weekday_off_peak": ("200_weekdays_10_11", "weekdays", (10, 11)),
    # "weekend_on_peak": ("200_weekends_12_19", "weekends", (12, 19)),
    # "weekend_off_peak": ("200_weekends_9_20", "weekends", (9, 20)),
}
# The remaining saved trainings are selectable explicitly, but were produced
# with older runtime settings and therefore fail the compatibility preflight
# unless that mismatch is intentionally overridden.
DEFAULT_MODELS = ("weekday_on_peak",)

PAIR_EXPERIMENT_KEYS = ("focus_hours", "day_filter", "selected_taz_id", "global_seed")
PAIR_LOCAL_CONTROLLER_KEYS = ("action_bins",)
PAIR_STRUCTURE_KEYS = ("taz_ids", "tls_ids", "control_groups_by_taz")

RUNTIME_DEMAND_KEYS = (
    "base_demand",
    "hourly_profile",
    "weekend_demand_factor",
    "weekend_hourly_profile",
    "demand_noise_range",
    "random_trip_seed",
    "route_sampler_seed",
    "route_sampler_threads",
    "vary_route_seeds_by_episode",
)
RUNTIME_COORDINATOR_KEYS = (
    "taz_adjacency_distance_threshold",
    "taz_adjacency_fallback_k",
)


def _checkpoint(training: str, controller: str) -> str:
    if training == "__current__":
        return os.path.join(
            PACKAGE_DIR,
            "checkpoints",
            controller,
            f"{controller}_best.pt",
        )
    checkpoint_path = os.path.join(
        SAVED_TRAININGS_DIR,
        training,
        "checkpoints",
        controller,
        f"{controller}_best.pt",
    )
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return os.path.join(
        SAVED_TRAININGS_DIR,
        training,
        controller,
        f"{controller}_best.pt",
    )


def _comparable(value: Any) -> Any:
    """Normalize list/tuple differences commonly introduced by serialization."""

    if isinstance(value, Mapping):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_comparable(item) for item in value)
    return value


def _checkpoint_local_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("local_controller_state_dict", "model_state_dict", "local_model_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, Mapping):
            return state_dict
    return {}


def _append_section_mismatches(
    mismatches: list[str],
    section_name: str,
    local_section: Mapping[str, Any],
    coordinated_section: Mapping[str, Any],
    keys: Sequence[str],
) -> None:
    for key in keys:
        local_value = _comparable(local_section.get(key))
        coordinated_value = _comparable(coordinated_section.get(key))
        if local_value != coordinated_value:
            mismatches.append(
                f"{section_name}.{key}: local={local_section.get(key)!r}, "
                f"coordinated={coordinated_section.get(key)!r}"
            )


def _append_runtime_mismatches(
    mismatches: list[str],
    section_name: str,
    checkpoint_section: Mapping[str, Any],
    runtime_section: Mapping[str, Any],
    keys: Sequence[str],
) -> None:
    for key in keys:
        checkpoint_value = _comparable(checkpoint_section.get(key))
        runtime_value = _comparable(runtime_section.get(key))
        if checkpoint_value != runtime_value:
            mismatches.append(
                f"runtime.{section_name}.{key}: checkpoint={checkpoint_section.get(key)!r}, "
                f"current={runtime_section.get(key)!r}"
            )


def _local_base_weight_mismatch(local: Mapping[str, Any], coordinated: Mapping[str, Any]) -> str | None:
    """Return a compact diagnostic when the coordinated base is not the saved local policy."""

    import torch

    local_state = _checkpoint_local_state_dict(local)
    coordinated_state = _checkpoint_local_state_dict(coordinated)
    if not local_state:
        return "local checkpoint has no local-controller state_dict"
    if not coordinated_state:
        return "coordinated checkpoint has no local-controller state_dict"

    policy_class = str(coordinated.get("local_policy_class", "MultiDiscreteActorCritic"))
    prefix = "base_policy." if policy_class == "PriceAdapterActorCritic" else ""
    missing_keys = []
    shape_mismatches = []
    changed_keys = []
    max_abs_difference = 0.0
    for key, local_tensor in local_state.items():
        coordinated_key = f"{prefix}{key}"
        coordinated_tensor = coordinated_state.get(coordinated_key)
        if coordinated_tensor is None:
            missing_keys.append(coordinated_key)
            continue
        if not isinstance(local_tensor, torch.Tensor) or not isinstance(coordinated_tensor, torch.Tensor):
            if local_tensor != coordinated_tensor:
                changed_keys.append(coordinated_key)
            continue
        if tuple(local_tensor.shape) != tuple(coordinated_tensor.shape):
            shape_mismatches.append(
                f"{coordinated_key} ({tuple(local_tensor.shape)} != {tuple(coordinated_tensor.shape)})"
            )
            continue
        local_cpu = local_tensor.detach().cpu()
        coordinated_cpu = coordinated_tensor.detach().cpu()
        if not torch.equal(local_cpu, coordinated_cpu):
            changed_keys.append(coordinated_key)
            if local_cpu.numel() > 0:
                max_abs_difference = max(
                    max_abs_difference,
                    float((local_cpu - coordinated_cpu).abs().max().item()),
                )

    expected_coordinated_keys = {f"{prefix}{key}" for key in local_state}
    if prefix:
        actual_base_keys = {key for key in coordinated_state if str(key).startswith(prefix)}
    else:
        actual_base_keys = set(coordinated_state)
    unexpected_keys = sorted(actual_base_keys - expected_coordinated_keys)

    diagnostics = []
    if missing_keys:
        diagnostics.append(f"missing={missing_keys[:5]!r}")
    if unexpected_keys:
        diagnostics.append(f"unexpected={unexpected_keys[:5]!r}")
    if shape_mismatches:
        diagnostics.append(f"shape={shape_mismatches[:5]!r}")
    if changed_keys:
        diagnostics.append(
            f"changed={changed_keys[:5]!r}, max_abs_difference={max_abs_difference:.6g}"
        )
    if diagnostics:
        return "; ".join(diagnostics)
    return None


def _checkpoint_state_errors(
    local: Mapping[str, Any],
    coordinated: Mapping[str, Any],
) -> list[str]:
    """Load state dicts into metadata-derived policies during the preflight."""

    import torch
    from rl_module.policy import MultiDiscreteActorCritic, PriceAdapterActorCritic

    errors = []
    local_state = _checkpoint_local_state_dict(local)
    coordinated_local_state = _checkpoint_local_state_dict(coordinated)
    coordinator_state = coordinated.get(
        "coordinator_state_dict",
        coordinated.get("global_model_state_dict"),
    )
    if not isinstance(coordinator_state, Mapping):
        errors.append("coordinated checkpoint has no coordinator_state_dict")
        return errors

    for label, state_dict in (
        ("local", local_state),
        ("coordinated local", coordinated_local_state),
        ("coordinator", coordinator_state),
    ):
        if not state_dict:
            errors.append(f"{label} state_dict is empty")
            continue
        non_finite = [
            str(key)
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
            and value.numel() > 0
            and not bool(torch.isfinite(value).all())
        ]
        if non_finite:
            errors.append(f"{label} state_dict has non-finite tensors: {non_finite[:5]!r}")

    local_settings = dict(local.get("local_controller_settings", {}) or {})
    local_action_bins = tuple(local_settings.get("action_bins", (-10.0, 0.0, 10.0)))
    local_obs_dim = local.get("obs_dim")
    local_act_dim = local.get("act_dim")
    local_logits = local_state.get("logits_head.weight")
    if local_act_dim is None and isinstance(local_logits, torch.Tensor) and local_action_bins:
        local_act_dim = int(local_logits.shape[0]) // len(local_action_bins)
    try:
        local_policy = MultiDiscreteActorCritic(
            int(local_obs_dim),
            int(local_act_dim),
            local_action_bins,
        )
        local_policy.load_state_dict(local_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"local state/shape validation failed: {str(exc).splitlines()[0]}")

    coordinator_settings = dict(coordinated.get("coordinator_settings", {}) or {})
    price_bins = tuple(coordinator_settings.get("price_bins", (0.0, 0.5, 1.0, 1.5, 2.0)))
    taz_ids = tuple(coordinated.get("taz_ids", ()) or ())
    coordinator_obs_dim = coordinated.get("coordinator_obs_dim")
    try:
        coordinator_policy = MultiDiscreteActorCritic(
            int(coordinator_obs_dim),
            len(taz_ids),
            price_bins,
        )
        coordinator_policy.load_state_dict(coordinator_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"coordinator state/shape validation failed: {str(exc).splitlines()[0]}")

    coordinated_local_settings = dict(coordinated.get("local_controller_settings", {}) or {})
    coordinated_action_bins = tuple(
        coordinated_local_settings.get("action_bins", local_action_bins)
    )
    policy_class = str(coordinated.get("local_policy_class", "MultiDiscreteActorCritic"))
    try:
        if policy_class == "PriceAdapterActorCritic":
            base_obs_dim = int(coordinated.get("base_obs_dim"))
            price_context_dim = int(
                coordinator_settings.get(
                    "local_price_context_dim",
                    int(coordinated.get("augmented_obs_dim")) - base_obs_dim,
                )
            )
            base_logits = coordinated_local_state.get("base_policy.logits_head.weight")
            coordinated_act_dim = (
                int(base_logits.shape[0]) // len(coordinated_action_bins)
                if isinstance(base_logits, torch.Tensor) and coordinated_action_bins
                else int(local_act_dim)
            )
            coordinated_local_policy = PriceAdapterActorCritic(
                base_obs_dim=base_obs_dim,
                price_context_dim=price_context_dim,
                act_dim=coordinated_act_dim,
                action_bins=coordinated_action_bins,
                adapter_hidden_dim=int(coordinator_settings.get("price_adapter_hidden_dim", 64)),
                adapter_scale=float(coordinator_settings.get("price_adapter_scale", 0.25)),
                adapter_mode=str(coordinator_settings.get("price_adapter_mode", "legacy")),
            )
        elif policy_class == "MultiDiscreteActorCritic":
            logits = coordinated_local_state.get("logits_head.weight")
            coordinated_act_dim = (
                int(logits.shape[0]) // len(coordinated_action_bins)
                if isinstance(logits, torch.Tensor) and coordinated_action_bins
                else int(local_act_dim)
            )
            coordinated_local_policy = MultiDiscreteActorCritic(
                int(coordinated.get("augmented_obs_dim", coordinated.get("base_obs_dim"))),
                coordinated_act_dim,
                coordinated_action_bins,
            )
        else:
            raise ValueError(f"unsupported local_policy_class={policy_class!r}")
        coordinated_local_policy.load_state_dict(coordinated_local_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(
            "coordinated local state/shape validation failed: "
            f"{str(exc).splitlines()[0]}"
        )
    return errors


def _validate_checkpoint_pair(
    model_name: str,
    day_filter: str,
    hours: Sequence[int],
    local_path: str,
    coordinated_path: str,
) -> tuple[list[str], list[str]]:
    import torch
    from rl_module.settings import COORDINATOR, DEMAND, ENVIRONMENT

    local = torch.load(local_path, map_location="cpu", weights_only=True)
    coordinated = torch.load(coordinated_path, map_location="cpu", weights_only=True)
    local_experiment = dict(local.get("experiment_settings", {}) or {})
    coordinated_experiment = dict(coordinated.get("experiment_settings", {}) or {})
    local_environment = dict(local.get("environment_settings", {}) or {})
    coordinated_environment = dict(coordinated.get("environment_settings", {}) or {})
    local_controller = dict(local.get("local_controller_settings", {}) or {})
    coordinated_local_controller = dict(coordinated.get("local_controller_settings", {}) or {})
    local_demand = dict(local.get("demand_settings", {}) or {})
    coordinated_demand = dict(coordinated.get("demand_settings", {}) or {})
    coordinator_settings = dict(coordinated.get("coordinator_settings", {}) or {})
    mismatches = []
    warnings = []

    _append_section_mismatches(
        mismatches,
        "experiment",
        local_experiment,
        coordinated_experiment,
        PAIR_EXPERIMENT_KEYS,
    )
    _append_section_mismatches(
        mismatches,
        "environment",
        local_environment,
        coordinated_environment,
        tuple(sorted(set(local_environment) | set(coordinated_environment))),
    )
    _append_section_mismatches(
        mismatches,
        "local_controller",
        local_controller,
        coordinated_local_controller,
        PAIR_LOCAL_CONTROLLER_KEYS,
    )
    _append_section_mismatches(
        mismatches,
        "demand",
        local_demand,
        coordinated_demand,
        RUNTIME_DEMAND_KEYS,
    )
    _append_section_mismatches(
        mismatches,
        "structure",
        local,
        coordinated,
        PAIR_STRUCTURE_KEYS,
    )

    requested_hours = tuple(int(hour) for hour in hours)
    for checkpoint_name, experiment in (
        ("local", local_experiment),
        ("coordinated", coordinated_experiment),
    ):
        checkpoint_hours = _comparable(experiment.get("focus_hours"))
        if checkpoint_hours != requested_hours:
            mismatches.append(
                f"requested focus_hours={requested_hours!r}, "
                f"{checkpoint_name} checkpoint focus_hours={experiment.get('focus_hours')!r}"
            )
        if experiment.get("day_filter") != day_filter:
            mismatches.append(
                f"requested day_filter={day_filter!r}, "
                f"{checkpoint_name} checkpoint day_filter={experiment.get('day_filter')!r}"
            )

    if local.get("architecture_name") not in (None, "local_controller"):
        mismatches.append(
            f"local architecture_name={local.get('architecture_name')!r}, expected 'local_controller'"
        )
    if coordinated.get("architecture_name") not in (None, "coordinated_controller"):
        mismatches.append(
            "coordinated architecture_name="
            f"{coordinated.get('architecture_name')!r}, expected 'coordinated_controller'"
        )

    local_obs_dim = local.get("obs_dim")
    coordinated_base_obs_dim = coordinated.get("base_obs_dim")
    if (
        local_obs_dim is not None
        and coordinated_base_obs_dim is not None
        and int(local_obs_dim) != int(coordinated_base_obs_dim)
    ):
        mismatches.append(
            f"observation width: local={local_obs_dim!r}, "
            f"coordinated base={coordinated_base_obs_dim!r}"
        )

    base_weight_mismatch = _local_base_weight_mismatch(local, coordinated)
    if base_weight_mismatch:
        mismatches.append(f"local base weights: {base_weight_mismatch}")
    mismatches.extend(_checkpoint_state_errors(local, coordinated))

    _append_runtime_mismatches(
        warnings,
        "demand",
        local_demand,
        dict(DEMAND.__dict__),
        RUNTIME_DEMAND_KEYS,
    )
    _append_runtime_mismatches(
        mismatches,
        "environment",
        local_environment,
        dict(ENVIRONMENT.__dict__),
        tuple(ENVIRONMENT.__dict__),
    )
    _append_runtime_mismatches(
        mismatches,
        "coordinator",
        coordinator_settings,
        dict(COORDINATOR.__dict__),
        RUNTIME_COORDINATOR_KEYS,
    )

    if (
        bool(coordinated_local_controller.get("train_local_policy", False))
        or not bool(coordinated_local_controller.get("train_price_adapter", False))
    ):
        print(
            f"[WARN] {model_name}: coordinated checkpoint was trained with "
            f"train_local_policy={coordinated_local_controller.get('train_local_policy')} and "
            f"train_price_adapter={coordinated_local_controller.get('train_price_adapter')}. "
            "The exact base-weight audit above remains authoritative."
        )
    return mismatches, warnings


def _non_overridable_mismatch(message: str) -> bool:
    """State/architecture errors cannot be made safe by a CLI override."""

    return str(message).startswith((
        "structure.",
        "local_controller.action_bins",
        "local architecture_name",
        "coordinated architecture_name",
        "observation width",
        "local base weights",
        "local checkpoint",
        "coordinated checkpoint",
        "local state_dict",
        "coordinated local state_dict",
        "coordinator state_dict",
        "local state/shape",
        "coordinated local state/shape",
        "coordinator state/shape",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SPECS,
        default=list(DEFAULT_MODELS),
        help=(
            "Saved training profiles to evaluate. The default is the runtime-compatible "
            f"profile: {', '.join(DEFAULT_MODELS)}."
        ),
    )
    parser.add_argument("--episodes-per-timeslot", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--stochastic-runs",
        type=int,
        default=0,
        help="Sampled-policy runs per scenario in addition to the deterministic greedy run.",
    )
    parser.add_argument(
        "--report-root",
        default=None,
    )
    parser.add_argument(
        "--demand-noise",
        type=float,
        default=None,
        help=(
            "Optional fixed demand multiplier passed to compare_coordinated_controller.py. "
            "Use values below 1.0 to test fewer vehicles while keeping the same date/hour route setup."
        ),
    )
    parser.add_argument("--keep-raw-output", action="store_true")
    parser.add_argument(
        "--force-rerun",
        "--force",
        "--no-resume",
        dest="force_rerun",
        action="store_true",
        help="Re-run every scenario even when a completed report exists.",
    )
    parser.add_argument(
        "--allow-incompatible-checkpoints",
        action="store_true",
        help=(
            "Override intentional metadata/runtime mismatches. Architecture, state, "
            "structure, and local-base-weight errors always remain fatal."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoints and print commands without starting SUMO.",
    )
    args = parser.parse_args()

    if args.episodes_per_timeslot < 1:
        raise ValueError("--episodes-per-timeslot must be >= 1")
    if args.stochastic_runs < 0:
        raise ValueError("--stochastic-runs must be >= 0")
    if args.demand_noise is not None and args.demand_noise <= 0.0:
        raise ValueError("--demand-noise must be > 0")

    if args.report_root is None:
        report_root = os.path.join(PACKAGE_DIR, "evaluation_reports", "saved_models_2026")
        if args.demand_noise is not None:
            demand_tag = f"demand_noise_{args.demand_noise:.4f}".replace(".", "p")
            report_root = os.path.join(PACKAGE_DIR, "evaluation_reports", f"saved_models_2026_{demand_tag}")
    else:
        report_root = os.path.abspath(args.report_root)

    selected_models = []
    for model_name in args.models:
        training, day_filter, hours = MODEL_SPECS[model_name]
        coordinated = _checkpoint(training, "coordinated_controller")
        local = _checkpoint(training, "local_controller")
        missing = [path for path in (coordinated, local) if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints for {model_name}: {', '.join(missing)}")
        mismatches, compatibility_warnings = _validate_checkpoint_pair(
            model_name,
            day_filter,
            hours,
            local,
            coordinated,
        )
        if compatibility_warnings:
            print(
                f"[WARN] {model_name}: evaluation demand differs from training "
                "(valid out-of-distribution test):\n- "
                + "\n- ".join(compatibility_warnings)
            )
        fatal_mismatches = [
            mismatch for mismatch in mismatches if _non_overridable_mismatch(mismatch)
        ]
        overridable_mismatches = [
            mismatch for mismatch in mismatches if mismatch not in fatal_mismatches
        ]
        if fatal_mismatches:
            raise ValueError(
                f"Invalid checkpoint state/structure for {model_name}; "
                "these errors cannot be overridden:\n- "
                + "\n- ".join(fatal_mismatches)
            )
        if overridable_mismatches and not args.allow_incompatible_checkpoints:
            raise ValueError(
                f"Incompatible checkpoints for {model_name}. "
                "Use --allow-incompatible-checkpoints only if this comparison is intentional:\n- "
                + "\n- ".join(overridable_mismatches)
            )
        if overridable_mismatches:
            print(
                f"[WARN] {model_name}: incompatible checkpoint metadata:\n- "
                + "\n- ".join(overridable_mismatches)
            )
        selected_models.append((model_name, training, day_filter, hours, coordinated, local))

    for model_name, training, day_filter, hours, coordinated, local in selected_models:
        command = [
            sys.executable,
            COMPARE_SCRIPT,
            "--start-date", "2026-01-01",
            "--end-date", "2026-12-31",
            "--day-filter", day_filter,
            "--timeslots", ",".join(f"{hour:02d}:00-{(hour + 1) % 24:02d}:00" for hour in hours),
            "--random-scenarios-per-timeslot", str(args.episodes_per_timeslot),
            "--random-seed", str(args.random_seed),
            "--checkpoint", coordinated,
            "--local-checkpoint", local,
            "--validation-name", model_name,
            "--report-root", report_root,
            "--stochastic-runs", str(args.stochastic_runs),
        ]
        if args.demand_noise is not None:
            command.extend(["--demand-noise", str(args.demand_noise)])
        if not args.force_rerun:
            command.append("--resume")
        if args.keep_raw_output:
            command.append("--keep-raw-output")

        print(
            f"\n[MODEL] {model_name} | training={training} | "
            f"episodes={args.episodes_per_timeslot * len(hours)} | "
            f"stochastic_runs={args.stochastic_runs}",
            flush=True,
        )
        if args.dry_run:
            print(f"[DRY RUN] {shlex.join(command)}", flush=True)
            continue
        subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()

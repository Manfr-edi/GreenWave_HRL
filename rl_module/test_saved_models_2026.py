from __future__ import annotations

"""Run the requested 2026 random-date evaluations for saved RL trainings."""

import argparse
import os
import subprocess
import sys


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARE_SCRIPT = os.path.join(PACKAGE_DIR, "compare_coordinated_controller.py")
SAVED_TRAININGS_DIR = os.path.join(PACKAGE_DIR, "saved_trainings")

MODEL_SPECS = {
    "weekday_on_peak": ("200_weekdays_8_17", "weekdays", (8, 17)),
    # "weekday_off_peak": ("200_weekdays_10_11", "weekdays", (10, 11)),
    # "weekend_on_peak": ("200_weekends_12_19", "weekends", (12, 19)),
    # "weekend_off_peak": ("200_weekends_9_20", "weekends", (9, 20)),
}


def _checkpoint(training: str, controller: str) -> str:
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


def _validate_checkpoint_pair(model_name: str, day_filter: str, local_path: str, coordinated_path: str) -> list[str]:
    import torch

    local = torch.load(local_path, map_location="cpu", weights_only=True)
    coordinated = torch.load(coordinated_path, map_location="cpu", weights_only=True)
    local_experiment = dict(local.get("experiment_settings", {}) or {})
    coordinated_experiment = dict(coordinated.get("experiment_settings", {}) or {})
    local_environment = dict(local.get("environment_settings", {}) or {})
    coordinated_environment = dict(coordinated.get("environment_settings", {}) or {})
    mismatches = []
    for key in ("focus_hours", "day_filter"):
        if local_experiment.get(key) != coordinated_experiment.get(key):
            mismatches.append(
                f"experiment.{key}: local={local_experiment.get(key)!r}, "
                f"coordinated={coordinated_experiment.get(key)!r}"
            )
    for key in ("min_green", "max_green", "cooldownSteps"):
        if local_environment.get(key) != coordinated_environment.get(key):
            mismatches.append(
                f"environment.{key}: local={local_environment.get(key)!r}, "
                f"coordinated={coordinated_environment.get(key)!r}"
            )
    if local_experiment.get("day_filter") != day_filter:
        mismatches.append(
            f"requested day_filter={day_filter!r}, checkpoint day_filter={local_experiment.get('day_filter')!r}"
        )

    local_settings = dict(coordinated.get("local_controller_settings", {}) or {})
    if bool(local_settings.get("train_local_policy", False)) or not bool(local_settings.get("train_price_adapter", False)):
        print(
            f"[WARN] {model_name}: coordinated checkpoint was trained with "
            f"train_local_policy={local_settings.get('train_local_policy')} and "
            f"train_price_adapter={local_settings.get('train_price_adapter')}; "
            "local-vs-coordinated is not a pure coordinator ablation."
        )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--episodes-per-timeslot", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
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
        help="Run even when local and coordinated checkpoint training metadata differ.",
    )
    args = parser.parse_args()

    if args.episodes_per_timeslot < 1:
        raise ValueError("--episodes-per-timeslot must be >= 1")
    if args.demand_noise is not None and args.demand_noise <= 0.0:
        raise ValueError("--demand-noise must be > 0")

    if args.report_root is None:
        report_root = os.path.join(PACKAGE_DIR, "evaluation_reports", "saved_models_2026")
        if args.demand_noise is not None:
            demand_tag = f"demand_noise_{args.demand_noise:.4f}".replace(".", "p")
            report_root = os.path.join(PACKAGE_DIR, "evaluation_reports", f"saved_models_2026_{demand_tag}")
    else:
        report_root = args.report_root

    selected_models = []
    for model_name in args.models:
        training, day_filter, hours = MODEL_SPECS[model_name]
        coordinated = _checkpoint(training, "coordinated_controller")
        local = _checkpoint(training, "local_controller")
        missing = [path for path in (coordinated, local) if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints for {model_name}: {', '.join(missing)}")
        mismatches = _validate_checkpoint_pair(model_name, day_filter, local, coordinated)
        if mismatches and not args.allow_incompatible_checkpoints:
            raise ValueError(
                f"Incompatible checkpoints for {model_name}. "
                "Use --allow-incompatible-checkpoints only if this comparison is intentional:\n- "
                + "\n- ".join(mismatches)
            )
        if mismatches:
            print(f"[WARN] {model_name}: incompatible checkpoint metadata:\n- " + "\n- ".join(mismatches))
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
            "--stochastic-runs", "0",
        ]
        if args.demand_noise is not None:
            command.extend(["--demand-noise", str(args.demand_noise)])
        if not args.force_rerun:
            command.append("--resume")
        if args.keep_raw_output:
            command.append("--keep-raw-output")

        print(f"\n[MODEL] {model_name} | training={training} | episodes={args.episodes_per_timeslot * len(hours)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

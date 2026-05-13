# RL Module

This module contains the cleaned-up RL stack for SUMO traffic-light control.
The module is organized around component roles:

- `local_controller`: per-TAZ traffic-light policy.
- `coordinator`: global policy that emits one coordination price per TAZ.
- `coordinated_controller`: full architecture composed of local controller plus coordinator.

## Files

- `settings.py`: central configuration for experiment length, day filters, demand noise, environment parameters, PPO hyperparameters, and output paths.
- `traffic_demand.py`: builds the training schedule, applies weekday filters, samples reproducible demand noise, and generates or reuses SUMO route caches.
- `environment.py`: defines the SUMO environment stack (`LocalTazTrafficEnv`, `GroupedTazTrafficEnv`, `CoordinatedTazTrafficEnv`) including TAZ observations, grouped TLS control, rewards, and TAZ-to-TAZ flow tracking.
- `policy.py`: defines the shared multi-discrete actor-critic used by both the local controller and the coordinator.
- `ppo.py`: contains PPO utilities, GAE computation, advantage normalization, entropy scheduling, seed setup, and optimizer-state helpers.
- `coordination.py`: builds TAZ adjacency, coordinator observations, local price features, and coordination reward shaping.
- `checkpoints.py`: contains checkpoint loading, local-pretraining weight loading, resume helpers, and CSV/JSON history helpers.
- `train_local_controller.py`: trains only the local per-TAZ traffic-light controller, without the global coordinator.
- `train_coordinated_controller.py`: trains the coordinated architecture using a local controller plus global coordinator.
- `compare_coordinated_controller.py`: compares baseline, local-only RL, and coordinated RL on the same SUMO route scenario.

## Schedule And Demand

`train_days` in `settings.py` means target number of training episodes, not a
calendar-day window. With:

```python
train_days = 200
focus_hours = (7, 8)
```

the training schedule contains exactly 200 date/hour episodes.

Supported `day_filter` values:

- `all`
- `weekdays`
- `weekends`
- explicit weekday lists such as `lun-mer-sab`

Italian and English weekday aliases are accepted, for example `lun`, `mer`,
`sab`, `mon`, `wed`, and `sat`.
Dates without traffic-flow rows in the processed CSV are skipped automatically;
the schedule continues scanning forward until it reaches `train_days`.

Demand is randomized but reproducible. The base seed remains `42`, and demand
noise is deterministically derived from the seed, episode index, date, and hour.
Saturday and Sunday episodes also apply `weekend_demand_factor` from
`settings.py`; if `weekend_hourly_profile` is configured, the weekend uses that
hourly profile before applying the attenuation factor.
Generated route caches are written under:

```text
sumoenv/routes_rl_module_randomized/
```

Each route path keeps date and time slot as folders, then uses one compact
scenario folder with day type, vehicle count, demand-noise tag, weekend factor,
base seed, and route seeds, so scenarios are explicit and reproducible without
deeply nested paths.

## Usage

Run commands from the standalone folder root with the `bomodt` environment active:

```bash
conda activate bomodt
```

### 1. Inspect The Schedule

Before launching SUMO, inspect the randomized schedule:

```bash
python rl_module/train_local_controller.py \
  --dry-run-schedule \
  --show-episodes 10
```

The coordinated trainer uses the same schedule logic:

```bash
python rl_module/train_coordinated_controller.py \
  --dry-run-schedule \
  --show-episodes 10
```

### 2. Train Local Controllers Only

Run full local-only training:

```bash
python rl_module/train_local_controller.py
```

Run a short debug training:

```bash
python rl_module/train_local_controller.py --max-episodes 2
```

Local-only outputs are written to:

```text
rl_module/checkpoints/local_controller/
rl_module/history/local_controller.csv
rl_module/history/local_controller.json
rl_module/details/local_controller/
```

The main checkpoint used by later coordinated training is:

```text
rl_module/checkpoints/local_controller/local_controller_best.pt
```

The best checkpoint is selected by `delta_penalty_vs_baseline`
(`rl_penalty - baseline_penalty`) when the no-agent baseline is available, so
low-demand episodes are not preferred only because their absolute penalty is low.

### 3. Train The Coordinated Controller

Run coordinated training:

```bash
python rl_module/train_coordinated_controller.py
```

Run a short debug training:

```bash
python rl_module/train_coordinated_controller.py --max-episodes 2
```

The coordinated trainer tries to initialize the local part from:

```text
rl_module/checkpoints/local_controller/local_controller_best.pt
```

If that file does not exist, the script prints a warning and starts the local
controller from random weights.

Coordinated outputs are written to:

```text
rl_module/checkpoints/coordinated_controller/
rl_module/history/coordinated_controller.csv
rl_module/history/coordinated_controller.json
rl_module/details/coordinated_controller/
```

### 4. Compare Baseline, Local RL, And Coordinated RL

Run the full comparison when both checkpoints are available:

```bash
python rl_module/compare_coordinated_controller.py \
  --date 2024-10-10 \
  --timeslot 08:00-09:00 \
  --local-checkpoint rl_module/checkpoints/local_controller/local_controller_best.pt \
  --checkpoint rl_module/checkpoints/coordinated_controller/coordinated_controller_best.pt \
  --stochastic-runs 3
```

Run only baseline vs local RL:

```bash
python rl_module/compare_coordinated_controller.py \
  --date 2024-10-10 \
  --timeslot 08:00-09:00 \
  --skip-coordinated \
  --stochastic-runs 3
```

Run only baseline vs coordinated RL:

```bash
python rl_module/compare_coordinated_controller.py \
  --date 2024-10-10 \
  --timeslot 08:00-09:00 \
  --skip-local \
  --stochastic-runs 3
```

Run validation over a broader calendar set. With `--month`, the script evaluates
all available traffic-flow dates in that month and uses `settings.py`
`focus_hours` unless `--timeslot` or `--timeslots` is provided:

```bash
python rl_module/compare_coordinated_controller.py \
  --month 2025-10 \
  --day-filter all \
  --stochastic-runs 3
```

Multiple months can be evaluated with `--months`:

```bash
python rl_module/compare_coordinated_controller.py \
  --months 2026-01,2026-02 \
  --day-filter all \
  --checkpoint-mode by-day-type \
  --stochastic-runs 3
```

For 2026 scenarios, route generation automatically uses:

```text
data/preprocessing/generated/processed_traffic_flow_2026.csv
```

Other years keep using the default processed traffic-flow file:

```text
data/preprocessing/generated/processed_traffic_flow.csv
```

By default, validation uses the same checkpoint for every scenario. To combine
the separate weekday and weekend trainings in one validation run, use:

```bash
python rl_module/compare_coordinated_controller.py \
  --month 2025-10 \
  --day-filter all \
  --checkpoint-mode by-day-type \
  --stochastic-runs 3
```

In this mode, weekday scenarios use:

```text
rl_module/saved_trainings/weekdays_7_8/checkpoints/local_controller/local_controller_best.pt
rl_module/saved_trainings/weekdays_7_8/checkpoints/coordinated_controller/coordinated_controller_best.pt
```

Weekend scenarios use:

```text
rl_module/saved_trainings/weekends_7_8/checkpoints/local_controller/local_controller_best.pt
rl_module/saved_trainings/weekends_7_8/checkpoints/coordinated_controller/coordinated_controller_best.pt
```

Override those defaults with `--weekday-checkpoint`, `--weekend-checkpoint`,
`--weekday-local-checkpoint`, and `--weekend-local-checkpoint` when needed.

For explicit slots over a range:

```bash
python rl_module/compare_coordinated_controller.py \
  --start-date 2025-10-01 \
  --end-date 2025-10-31 \
  --timeslots 07:00-08:00,08:00-09:00 \
  --stochastic-runs 3
```

Comparison reports are written under:

```text
rl_module/evaluation_reports/coordinated_controller/<date>/<timeslot>/<demand>/
```

Multi-scenario validation also writes:

```text
rl_module/evaluation_reports/coordinated_controller/validation/<validation-name>/validation_summary.json
rl_module/evaluation_reports/coordinated_controller/validation/<validation-name>/validation_summary.csv
```

The JSON report includes these sections when the corresponding run is enabled:

- `baseline`
- `local_greedy`
- `local_stochastic_mean`
- `coordinated_greedy`
- `coordinated_stochastic_mean`
- `comparison_local_greedy`
- `comparison_local_stochastic_mean`
- `comparison_coordinated_greedy`
- `comparison_coordinated_stochastic_mean`
- `comparison_coordinated_vs_local_greedy`
- `comparison_coordinated_vs_local_stochastic_mean`

Each run section also includes `emissions_by_taz` and `emission_taz_summary`.
Global totals still come from `tripinfos.xml`; per-TAZ emissions are aggregated
from `emission-output.xml` by assigning each vehicle sample to a TAZ polygon.
The CO2 summary reports the mean emission per active TAZ, the lowest-emission
TAZ, the highest-emission TAZ, and balance indicators (`range`, `std`, `cv`).
Lower balance indicators in the coordinated run compared with the local-only
run support the coordinator's TAZ-balancing effect.

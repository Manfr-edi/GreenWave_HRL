# BoMoDT RL Standalone

This repository is a self-contained extraction of the BoMoDT reinforcement
learning stack for SUMO traffic-light control. It can train either local
per-TAZ traffic-light policies or the coordinated architecture composed of the
local controller plus a global TAZ coordinator.

The standalone layout is designed to run without importing Python modules or
SUMO/data files from the original `BoMoDT` project.

## Tested Environment

The project has been tested with:

- Python `3.10`
- SUMO `1.26`
- Python packages installed from `requirements.txt`

SUMO must be installed separately and available through the `sumo` executable.
The route-generation utilities look for SUMO tools in this order:
`SUMO_TOOLS`, `SUMO_HOME/tools`, `/usr/share/sumo/tools`, then the Windows path
defined in `libraries/constants.py`.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

If the original conda environment is available, it can also be used:

```bash
conda activate bomodt
```

## Usage

Run every command from this standalone folder:

```bash
cd /home/stefano/PycharmProjects/BoMoDT_RL_Standalone
```

Inspect the deterministic randomized schedule used by the local trainer:

```bash
python rl_module/train_local_controller.py --dry-run-schedule --show-episodes 10
```

Train only the local agents:

```bash
python rl_module/train_local_controller.py
```

Train the coordinated controller:

```bash
python rl_module/train_coordinated_controller.py
```

Compare baseline, local-only RL, and coordinated RL:

```bash
python rl_module/compare_coordinated_controller.py \
  --date 2024-10-10 \
  --timeslot 08:00-09:00 \
  --local-checkpoint rl_module/checkpoints/local_controller/local_controller_best.pt \
  --checkpoint rl_module/checkpoints/coordinated_controller/coordinated_controller_best.pt \
  --stochastic-runs 3
```

Run validation on a full month, using the experiment focus hours by default:

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

For 2026 scenarios, route generation automatically uses
`data/preprocessing/generated/processed_traffic_flow_2026.csv`. Other years keep
using `data/preprocessing/generated/processed_traffic_flow.csv`.

Use weekday/weekend checkpoints together during the same validation run:

```bash
python rl_module/compare_coordinated_controller.py \
  --month 2025-10 \
  --day-filter all \
  --checkpoint-mode by-day-type \
  --stochastic-runs 3
```

In `by-day-type` mode, Monday-Friday scenarios use the checkpoints under
`rl_module/saved_trainings/weekdays_7_8/`, while Saturday-Sunday scenarios use
the checkpoints under `rl_module/saved_trainings/weekends_7_8/`. Pass
`--weekday-checkpoint`, `--weekend-checkpoint`, `--weekday-local-checkpoint`,
and `--weekend-local-checkpoint` to override those defaults.

Use explicit time slots for a range:

```bash
python rl_module/compare_coordinated_controller.py \
  --start-date 2025-10-01 \
  --end-date 2025-10-31 \
  --timeslots 07:00-08:00,08:00-09:00 \
  --stochastic-runs 3
```

For short smoke runs, both training scripts support `--max-episodes`:

```bash
python rl_module/train_local_controller.py --max-episodes 1
python rl_module/train_coordinated_controller.py --max-episodes 1
```

## RL Model Structure

The controller is implemented as a hierarchical reinforcement-learning system
with decentralized traffic-light control and an optional centralized
coordination layer. The local agent is executed once for each traffic analysis
zone (TAZ) and selects signal-duration updates for groups of nearby traffic
lights. The global agent, referred to as the coordinator, is executed once per
decision step and emits one coordination price per TAZ. These prices are not
applied directly to SUMO; they are appended to the local observation and used
to shape the coordinated reward.

Both agents use the `MultiDiscreteActorCritic` architecture in
`rl_module/policy.py`, but they keep separate parameter sets. The local policy
shares its weights across all TAZ rows, whereas the coordinator receives a
single flattened city-level observation.

```text
                       episode schedule and route demand
                                      |
                                      v
                                SUMO + TraCI
                                      |
              E2 detectors, vehicle states, TLS programs
                                      |
                                      v
                            CoordinatedTazTrafficEnv
                                      |
             +------------------------+------------------------+
             |                                                 |
             v                                                 v
    local TAZ observations                          TAZ adjacency and flow
    o_i = [TLS_i, TAZ_i, groups_i]                  city context
             |                                                 |
             |                               +-----------------+
             |                               |
             v                               v
    +---------------------+          +--------------------------+
    | Local agent         |          | Global coordinator       |
    | shared over TAZs    |<---------| one city-level policy    |
    +---------------------+ prices   +--------------------------+
             |                               |
             | group duration deltas         | price vector p_i
             v                               |
    SUMO traffic-light programs <------------+
             |
             v
    local rewards, coordinator reward, PPO/GAE updates
```

In local-only training, the same grouped local environment is used, but the
global coordinator is not instantiated and no price context is appended to the
local observation. In coordinated training, each decision step follows this
order: the coordinator observes the city state and samples prices, the price
features are appended to each TAZ observation, the local policy samples
duration updates, SUMO advances, and both local and coordinator rewards are
computed from the resulting traffic state.

### Shared Policy And Training Loop

`MultiDiscreteActorCritic` is used for both levels. It is a multi-discrete
actor-critic network with a feature extractor on top of the raw observation:

```text
observation
    |
    v
FeatureExtractor:
LayerNorm -> Linear(256) + GELU -> Linear(256) + GELU -> Linear(128) + GELU
    |
    +--> actor head: logits for each controlled slot and each action bin
    |
    +--> critic head: scalar value estimate
```

For the local agent, one observation row corresponds to one TAZ and the actor
produces one categorical distribution per valid control group in that TAZ. For
the coordinator, the input is a single flattened city vector and the actor
produces one categorical distribution per TAZ. Invalid or padded action slots
are masked and forced to the zero-action bin before sampling and evaluation.

The local controller is evaluated once per TAZ with shared weights. The global
coordinator is evaluated once per environment step with a single flattened city
observation. Both rollouts are updated with PPO using GAE, clipped policy loss,
value loss, entropy regularization, gradient clipping, and KL early stopping.

### Local Agent

The active trainers instantiate `CoordinatedTazTrafficEnv`, which extends
`GroupedTazTrafficEnv`. The local policy therefore controls topology-aware
groups of nearby traffic lights inside each TAZ, rather than individual raw TLS
slots. The lower-level `LocalTazTrafficEnv` still provides the per-TLS and
per-TAZ observation backbone, reward computation, and green-phase duration
editing logic.

**Observation.** Each local observation row represents one TAZ and is built from
live SUMO state. In compact form, the local observation for TAZ `i` is:

```text
o_i = [per-TLS features_i, TAZ aggregate features_i, group features_i, price context_i]
```

The final `price context_i` block is present only in coordinated training. The
non-price part contains:

- per-TLS blocks for all TLS slots in the TAZ, padded to the maximum TLS count:
  presence flag, current editable green phase as one-hot, normalized phase
  duration, mean speed, max jam, occupancy, E2 vehicle count, pressure, active
  vehicle count, total waiting time, and average waiting time;
- one TAZ aggregate block: max jam, mean speed, mean occupancy, detector
  vehicle total, active vehicles, total waiting time, hour
  sine/cosine, and deltas since the previous action for waiting time, active
  vehicles, jam, and speed;
- one grouped-control block per local group: presence, normalized group size,
  mean phase duration, mean active vehicles, mean waiting time, mean pressure,
  east-west green ratio, north-south green ratio, and dynamic signal.

In coordinated training, three additional price-context features are appended
to each TAZ row: own normalized coordinator price, neighbor mean price, and the
own-minus-neighbor price gap. The observation sources are E2 detector records,
TraCI vehicle metrics, traffic-light phase/program state, network topology,
and episode hour. Emission metrics are intentionally excluded from both local and
coordinator observations; they remain available only for reward computation and
evaluation. The runtime action mask removes padded groups and can
optionally disable groups whose traffic signal is below
`action_signal_threshold`.

The local feature extractor applies `LayerNorm` and the MLP described above to
this normalized, padded row. The resulting latent vector is shared by the actor,
which scores group-level duration actions, and the critic, which estimates the
value of the current TAZ state.

**Action.** The local policy emits one categorical action per valid control
group in the TAZ. The active action bins are:

```text
0 -> -10 seconds
1 ->   0 seconds
2 -> +10 seconds
```

The selected value is applied as a delta to editable green-phase durations.
The resulting duration is clipped to `min_green=25` and `max_green=120`
seconds. For multi-TLS groups, the environment can restrict the update to TLS
that currently share the dominant green axis (`EW` or `NS`), keeping the group
action aligned with the movement being served. The base per-TLS environment
also supports `-5, 0, +5` second bins, but the standalone training scripts use
the grouped `-10, 0, +10` setup.

**Reward.** The local reward is one scalar per TAZ. At each decision step the
environment builds a TAZ penalty from normalized waiting, emissions, and jam:

```text
penalty =
  waiting_weight  * waiting_penalty
+ emission_weight * (co2_mix * co2_penalty + nox_mix * nox_penalty)
+ jam_weight      * jam_penalty
```

The current `settings.py` defaults make the dense penalty emission-focused
(`waiting_weight=0.0`, `emission_weight=1.0`, `jam_weight=0.0`, with CO2 and
NOx mixed equally), but the terms remain configurable. The dense reward is:

```text
dense_reward =
  -dense_abs_penalty_weight * current_penalty
+ dense_delta_reward_weight * (previous_penalty - current_penalty)
```

It is clipped by `reward_clip`. At episode end, a terminal reward is added using
the terminal waiting/emission/jam penalty. If the no-agent baseline for the same
route/hour is available, the terminal reward is based on
`baseline_penalty - rl_penalty`; otherwise it is the negative absolute terminal
penalty. In coordinated training, the local reward can also be shaped by the
coordinator's local externality adjustment. Thus, the local policy is trained
to reduce its own TAZ penalty while remaining sensitive to spillover costs
estimated by the coordinator.

### Global Coordinator

The coordinator is active only in `train_coordinated_controller.py`. It runs
once per decision step before the local controller and emits one price per TAZ.
The prices act as soft coordination signals: they expose global congestion and
spillover information to local agents without directly overwriting local
traffic-light actions.

**Observation.** The coordinator receives one flattened city-level observation
built by `build_coordination_observation`:

- the 12-dimensional TAZ aggregate feature block extracted from every local TAZ
  observation;
- neighbor-mean TAZ features using the adjacency matrix built from TAZ polygons
  (`taz_adjacency_distance_threshold=80.0`, fallback to the nearest 3 TAZs when
  needed);
- previous TAZ-to-TAZ flow features: outgoing, incoming, neighbor outgoing, and
  neighbor incoming flow, normalized by `flow_ref`;
- previous coordinator prices: own price, neighbor mean price, and price gap;
- previous local action summary per TAZ: average applied duration delta,
  nonzero-update ratio, and aggressive-action ratio;
- city summaries over TAZ features: mean, standard deviation, and max;
- episode context: hour, vehicle count relative to base demand, demand noise,
  and baseline penalty when available.

TAZ flow is tracked online by remembering each active vehicle's last observed
TAZ and counting source-to-destination TAZ transitions after every SUMO step.
The coordinator feature extractor applies the same `LayerNorm` plus MLP stack
to this flattened city vector. Its actor head scores one price-bin action for
each TAZ, and its critic estimates the value of the city-level state.

**Action.** The coordinator action space has one categorical slot per TAZ. The
action bins are the coordination prices:

```text
0 -> 0.0
1 -> 0.5
2 -> 1.0
3 -> 1.5
4 -> 2.0
```

The resulting price vector is appended to the current local decision context as
own price, neighbor price, and price gap.

**Reward.** The coordinator reward is one scalar for the whole city. It is
computed after the local actions are applied and combines step-level traffic
improvement, terminal baseline comparison when available, spatial balance,
spillover/externality, price effort, and price-to-worsening alignment:

```text
global_reward =
  city_delta_weight    * sum(previous_penalty - current_penalty)
+ terminal_delta_weight * terminal_or_step_city_delta
- imbalance_weight     * std(current_taz_penalties)
- spillover_weight     * mean(externality)
- price_effort_weight  * mean(abs(normalized_price))
+ 0.10                 * price_worsening_alignment
```

The reward is clipped by `global_reward_clip`. The same computation also
produces a local adjustment:

```text
local_adjustment =
  -local_externality_weight * externality
+ local_global_share_weight * city_delta / num_taz
```

This adjustment is clipped by `local_bonus_clip` and added to the local reward
when the coordinated run is configured to keep training the local policy. In
practice, the coordinator is rewarded for improving the city-level penalty,
avoiding uneven TAZ penalties, discouraging downstream spillover, and using
prices only when they are useful.

## Generated Files

Training writes persistent experiment artifacts under `rl_module/` and
deterministic SUMO route caches under `sumoenv/routes_rl_module_randomized/`.
Route caches are shared by local and coordinated training when the episode
date, time slot, demand, and seeds match.

The examples below show the main files produced by a one-episode run.

### Local Training

Command:

```bash
python rl_module/train_local_controller.py --max-episodes 1
```

Generated artifacts:

```text
rl_module/
├── checkpoints/
│   └── local_controller/
│       ├── local_controller_episode0.pt      # Full checkpoint for episode 0.
│       ├── local_controller_best.pt          # Best rollout policy selected by checkpoint score.
│       └── local_controller_final.pt         # Final training state at the end of the run.
├── details/
│   └── local_controller/
│       └── episode_0000.json                 # Episode demand, baseline, terminal reward, and diagnostics.
└── history/
    ├── local_controller.csv                  # Compact per-episode metrics table.
    └── local_controller.json                 # Same training history serialized as JSON.

sumoenv/
└── routes_rl_module_randomized/
    └── <date>/
        └── <timeslot>/
            └── <scenario>/
                ├── generatedRoutes.rou.xml   # SUMO routes used by the episode.
                ├── randomTrips.rou.xml       # Random-trip generator output.
                └── trips.rou.xml             # Trip definitions before route generation.
```

The local history rows include the episode id, date, day type, time slot,
demand factors, vehicle count, terminal penalty, baseline penalty,
`delta_penalty_vs_baseline`, reward mean, PPO loss, learning rate, seeds, and
route-cache path.

### Coordinated Training

Command:

```bash
python rl_module/train_coordinated_controller.py --max-episodes 1
```

Generated artifacts:

```text
rl_module/
├── checkpoints/
│   └── coordinated_controller/
│       ├── coordinated_controller_episode0.pt # Full local-controller and coordinator checkpoint.
│       ├── coordinated_controller_best.pt     # Best coordinated rollout policy selected by checkpoint score.
│       └── coordinated_controller_final.pt    # Final training state at the end of the run.
├── details/
│   └── coordinated_controller/
│       └── episode_0000.json                  # Demand, baseline, coordination, flow, and adjacency diagnostics.
└── history/
    ├── coordinated_controller.csv             # Compact per-episode coordinated metrics table.
    └── coordinated_controller.json            # Same coordinated history serialized as JSON.

sumoenv/
└── routes_rl_module_randomized/
    └── <date>/
        └── <timeslot>/
            └── <scenario>/
                ├── generatedRoutes.rou.xml
                ├── randomTrips.rou.xml
                └── trips.rou.xml
```

The coordinated history extends the local metrics with coordinator-specific
values such as shaped local reward, coordination adjustment, coordinator reward,
coordinator price mean/std, spillover, imbalance, episode flow total, local PPO
loss, and coordinator PPO loss.

During training, SUMO also writes runtime XML files such as `tripinfos.xml`,
`emission-output.xml`, `summary.xml`, `vehroute.xml`, and
`edgedata-output.xml` under the active scenario output folder. The training
scripts clean those runtime files after each baseline and RL episode. Evaluation
runs keep their reports and copied SUMO outputs under
`rl_module/evaluation_reports/`.

## Repository Structure

```text
BoMoDT_RL_Standalone/
├── README.md                                 # Project overview, setup, usage, generated files, and structure.
├── requirements.txt                          # Minimal Python dependencies for the standalone RL module.
├── data/
│   ├── preprocessing/
│   │   └── generated/
│   │       ├── processed_traffic_flow.csv    # Default processed traffic-flow input used for route generation.
│   │       ├── 2024_processed_traffic_flow.csv # Year-specific processed traffic-flow input when available.
│   │       ├── processed_traffic_flow_2026.csv # Processed traffic-flow input used automatically for 2026 scenarios.
│   │       └── edgedata.xml                  # Edge-data file generated/used by preprocessing and SUMO workflows.
│   └── realworlddata/
│       └── taz_to_tls.json                   # Mapping from traffic analysis zones to traffic lights.
├── libraries/
│   ├── constants.py                          # Shared project paths for data, SUMO, routes, and generated files.
│   ├── classes/
│   │   ├── Planner.py                        # SUMO route and trip generation helpers.
│   │   └── SumoSimulator.py                  # SUMO/TraCI simulator wrapper used by the RL environments.
│   └── utils/
│       └── preprocessingUtils.py             # Preprocessing helpers kept from the BoMoDT data workflow.
├── rl_module/
│   ├── README.md                             # Detailed RL-module notes and command examples.
│   ├── settings.py                           # Experiment, demand, environment, local PPO, and coordinator settings.
│   ├── traffic_demand.py                     # Episode schedule, deterministic demand noise, and route-cache logic.
│   ├── environment.py                        # SUMO RL environments, rewards, TAZ metrics, and diagnostics.
│   ├── policy.py                             # Multi-discrete actor-critic policy shared by local and coordinator agents.
│   ├── ppo.py                                # PPO, GAE, entropy scheduling, seeding, and optimizer utilities.
│   ├── coordination.py                       # TAZ adjacency, price features, coordinator observations, and rewards.
│   ├── checkpoints.py                        # Checkpoint loading, resume helpers, and history writers.
│   ├── train_local_controller.py             # Local-only PPO training entry point.
│   ├── train_coordinated_controller.py       # Coordinated local-controller plus global-coordinator training entry point.
│   ├── compare_coordinated_controller.py     # Baseline/local/coordinated comparison and report generation.
│   ├── checkpoints/                          # Generated model checkpoints for local and coordinated runs.
│   ├── details/                              # Generated per-episode JSON diagnostics.
│   ├── history/                              # Generated CSV/JSON training histories.
│   ├── evaluation_reports/                   # Generated comparison reports and preserved SUMO outputs.
│   └── saved_trainings/                      # Optional archived training artifacts.
└── sumoenv/
    ├── standalone/
    │   ├── run.sumocfg                       # Base standalone SUMO configuration.
    │   └── run_rl.sumocfg                    # RL SUMO configuration with route, detector, and output paths.
    ├── static/
    │   ├── full.net.xml                      # SUMO network.
    │   ├── detectors.add.xml                 # Detector definitions.
    │   ├── e2Detector.xml                    # E2 detector configuration loaded by RL runs.
    │   ├── optimized_tls.add.xml             # Traffic-light logic/additional SUMO definitions.
    │   └── output_taz.add.xml                # TAZ polygons/additional file used for TAZ metrics.
    ├── routes_rl_module_randomized/          # Generated deterministic route caches by date, time slot, and demand.
    └── output/                               # Runtime SUMO output folder, normally empty after cleanup.
```

# BoMoDT RL Standalone

This folder is a self-contained extraction of the RL traffic-light module. It is
designed to run without importing Python files or SUMO/data files from the
original `BoMoDT` project.

## What Is Included

- `rl_module/`: local-only training, coordinated training, comparison scripts,
  PPO utilities, policies, settings, and best checkpoints.
- `taz_rl/rlenv/`: only the TAZ SUMO environment files required by
  `rl_module.environment`.
- `libraries/`: minimal constants, simulator wrapper, route planner, and
  preprocessing helper used by the RL scripts.
- `sumoenv/`: the SUMO network, detector/additional files, and SUMO
  configuration files needed by RL execution.
- `data/`: the processed traffic-flow CSV used to generate per-episode
  `edgedata.xml`, plus the TAZ-to-TLS mapping.

Generated route caches, histories, details, reports, and SUMO outputs are not
pre-copied. They are recreated inside this standalone folder when you run the
scripts.

## Requirements

Use the same `bomodt` conda environment if available:

```bash
conda activate bomodt
```

Otherwise install the minimal Python dependencies:

```bash
pip install -r requirements.txt
```

SUMO must be installed separately. The route planner looks for SUMO tools in
this order: `SUMO_TOOLS`, `SUMO_HOME/tools`, `/usr/share/sumo/tools`, then the
Windows path defined in `libraries/constants.py`.

## Usage

Run every command from this standalone folder:

```bash
cd /home/stefano/PycharmProjects/BoMoDT_RL_Standalone
```

Inspect the deterministic randomized schedule:

```bash
python rl_module/train_local_controller.py --dry-run-schedule --show-episodes 10
```

Train only local agents:

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

The scripts write new artifacts under `rl_module/` and route caches under
`sumoenv/routes_rl_module_randomized/`, all inside this standalone folder.

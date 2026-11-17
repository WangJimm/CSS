# CSS refactor

For the four imported RP models and setup on another Windows PC, start with
[README_PORTABLE_zh-TW.md](README_PORTABLE_zh-TW.md). The private repository includes
the old/new datasets and model weights; no separate RP checkout is required.

## Select the environment

Use `--env env` for the existing Brooklyn setup (the default), or `--env new_env`
for the Berlin dataset under this project's `new_env` folder. No config-file edit
is needed. The new environment reads `scenario_manifest.csv`: 420 training,
90 validation, and 90 final testing cases for demands 2650/2800/2950. Training
validation reads `validation_data`; final evaluation reads `testing_data`.
The existing `env` behavior and its historical seed ranges remain available.

Check all new inputs without starting SUMO, creating models, or training:

```powershell
python -m scripts.run_experiment --env new_env --check-data
```

Train the CS-selection agent using Dijkstra as the routing backend (requires no
RP checkpoint; it is a separate experimental setting from learned RP routing):

```powershell
python -m scripts.run_experiment --env new_env --agent sacd --route-mode dijkstra --mode train --episodes 1000
```

To use the learned RP proposed routing model, explicitly provide the model trained
for the new network. Replace the example path below with your actual checkpoint:

```powershell
python -m scripts.run_experiment --env new_env --agent sacd --route-mode dqngnn --mode train --episodes 1000 --route-checkpoint 'D:\models\new_env\gine_action_best_arrival.pt'
```

`--route-checkpoint` is the **routing** model. `--checkpoint` is the **charging
station selection** model. A Brooklyn route model is not a Berlin route model,
and new_env never falls back to a model stored under `route_planning_agent/DQN/env`.
Default new_env routing model locations are under `route_planning_agent/DQN/new_env`
(with the same per-backend filenames as env); an explicit `--route-checkpoint`
can point to any compatible RP checkpoint. GNN checkpoint graph dimensions are
checked, and checkpoints containing custom shape metadata support PyTorch 2.6+.

The stations come from `new_env/cs_locations.json`. Initial SOC comes from the
matching `*_data_power` lists and is mapped to actual EV IDs in XML order.
CS-selection agents still choose their stations at departure; fixed random goal
files are checked for alignment but do not override the agent's choice.
New_env reads the controlled traffic-light links, as RP does, so `GS_*` controller
IDs are handled even when they differ from junction IDs.

New_env outputs are isolated:

- Results: `outputs/new_env/`
- SUMO diagnostics: `logs/new_env/`
- TensorBoard: `runs/new_env/<agent>_route_<backend>/<timestamp>/`
- Default charging-selection checkpoint: `parameters/new_env/<agent>_route_<backend>.pt`

Use the same options with `--mode eval` to evaluate the saved charging agent on
all 90 testing cases; pass `--checkpoint` if you saved it to a custom location.
`cs_navigation.py` accepts the same options as the package entry point.


This directory is an independent refactoring copy of `D:\paper\CSS`. The
original project is read-only and remains unchanged.

## Preferred commands

Run the experiment through the package-aware entry point:

```powershell
$env:SUMO_HOME='D:\app\envs\osm\Lib\site-packages\sumo'
$env:SUMO_BINARY='D:\app\envs\osm\Lib\site-packages\sumo\bin\sumo.exe'
$env:PATH='D:\app\envs\osm\Lib\site-packages\sumo\bin;' + $env:PATH
& 'D:\app\envs\osm\python.exe' -m scripts.run_experiment --help
```

The historical command remains available:

```powershell
& 'D:\app\envs\osm\python.exe' cs_navigation.py --help
```

Standard evaluation is now the safe default because this checkout does not
contain the optional higher-evaluation dataset. Use `--higher-evaluation` only
after installing and auditing `env/higher_test_data` and
`env/higher_test_data_power`; the command fails before SUMO starts if either is
missing.

The trained route checkpoints currently available are `dqngnn`, `dqn`, and
`edqn`. The eDQN checkpoint was recovered from
`D:\paper\RP\outputs\edqn_baseline\tmp_file.pt`, copied without changing its
bytes, and verified with strict model and optimizer loading. It is the final
periodic checkpoint from episode 2499 of a 2500-episode run, not a separately
selected best-validation checkpoint. The CSS eDQN input adapter follows the RP
checkpoint's state schema: current-node one-hot, target-node one-hot, and one
normalized simulation-time value.

For example, run a charging-station agent with the recovered route model using:

```powershell
& 'D:\app\envs\osm\python.exe' -m scripts.run_experiment `
  --agent linucb --route-mode edqn --mode eval --standard-evaluation `
  --checkpoint .\contextual_linucb.pt
```

`--checkpoint` is the charging-station agent checkpoint. To select a different
route-planning checkpoint without replacing the default file, pass it separately
with `--route-checkpoint`. For example, train charging eDQN with the fine-tuned
route eDQN using:

```powershell
& 'D:\app\envs\osm\python.exe' -m scripts.run_experiment `
  --agent edqn --route-mode edqn --mode both --episodes 1000 `
  --training-start-episode 250 --standard-evaluation `
  --route-checkpoint '.\route_planning_agent\DQN\env\edqn_best_100pct_arrival_lowest_cost.pt' `
  --checkpoint '.\parameters\env_edqn_route_edqn_finetuned_ep1000.pt'
```

Run all tests in the experiment environment:

```powershell
& 'D:\app\envs\osm\python.exe' -m unittest discover -s tests -v
```

Run the deterministic old/new comparison:

```powershell
& 'D:\app\envs\osm\python.exe' scripts\compare_version_snapshot.py `
  --old 'D:\paper\CSS' `
  --new '.' `
  --python 'D:\app\envs\osm\python.exe' `
  --allow-difference schedule `
  --output 'comparison_results\version_snapshot.json'
```

`schedule` is the one intentional old/new difference after resolving CSS-001:
the refactor selects existing training seeds 1–20 instead of invalid 100–150.

Run the fixed LinUCB/Dijkstra simulation comparison without writing into the
original project. Execute `scripts/run_fixed_evaluation_cases.py` once with
`--project D:\paper\CSS` and once with `--project .`, using the same dataset,
seed list, checkpoint, and random seed. Then compare the two JSON files with:

```powershell
& 'D:\app\envs\osm\python.exe' scripts\compare_fixed_evaluation_cases.py `
  --original 'comparison_results\full_training_linucb_dijkstra_original.json' `
  --refactor 'comparison_results\full_training_linucb_dijkstra_refactor.json' `
  --output 'comparison_results\full_training_linucb_dijkstra_comparison.json' `
  --absolute-tolerance 1e-9
```

The included testing harness records aggregate metrics and each EV's selected
charging station, completion state, travel time, waiting time, and charging
time. Runtime-only fields and source paths are excluded from equivalence.

For actual learning-loop equivalence, `scripts/run_fixed_training_cases.py`
starts a fresh LinUCB agent, injects an explicit seed/probability schedule into
`DRL_train`, and records every EV action plus the LinUCB matrices after each
episode. All logs and checkpoints must point inside this refactor directory so
the original project remains read-only. Its JSON output can be compared with
the same comparison command above.

Run the short SUMO/TraCI/PyG integration smoke test:

```powershell
& 'D:\app\envs\osm\python.exe' -m scripts.smoke_sumo_package `
  --sumo 'D:\app\envs\osm\Lib\site-packages\sumo\bin\sumo.exe'
```

All issues found during the structural move have now been handled with
regression tests. `REFACTOR_ISSUES.md` records the prior behavior, resolution,
and validation policy for each one.

# CSS refactor architecture

## Source layout

```text
CSS_refactor/
├─ css/
│  ├─ agents/
│  │  ├─ charging/    charging-station selection agents
│  │  └─ routing/     next-junction route-planning agents
│  ├─ charging/       charging-station queue and state environment
│  ├─ experiment/     config, scenarios, planning, training, evaluation
│  ├─ graph/          PyTorch Geometric feature construction
│  ├─ routing/        topology and shortest-path algorithms
│  ├─ sumo/           TraCI, signals, distance, spawning, route XML
│  ├─ reporting/      plotting helpers
│  └─ legacy/         deprecated Dijkstra simulation
├─ scripts/           executable experiment and verification commands
├─ tests/             characterization, API, boundary, and CLI tests
├─ env/               SUMO network and scenario inputs
├─ parameters/        charging-selection model checkpoints
└─ *.py               compatibility shims and `utils_sumo.py` facade
```

## Dependency direction

```text
scripts
   ↓
css.experiment / css.charging / css.agents
   ↓
css.graph / css.sumo / css.routing
   ↓
SUMO, TraCI, PyTorch, PyTorch Geometric
```

Package and active script code must not import root compatibility modules.
`scripts.compare_version_snapshot` is the sole exception because its purpose is
to load both historical and new public paths in isolated subprocesses.

## Public imports

New code should import domain packages:

```python
from css.routing import dijkstra_path
from css.sumo import apply_full_route, remaining_routes_len
from css.graph import build_data_runtime, prepare_static
from css.charging import Env
from css.agents.charging import ContextualLinUCBAgent, DiscreteSACAgent
from css.agents.routing import route_DQNGNN
from css.experiment import ScenarioKey, build_training_scenario_schedule
```

Names beginning with `_` are implementation details and are intentionally not
included in package `__all__` lists.

## Compatibility policy

- `config.py`, `env.py`, `training.py`, and `cs_navigation.py` preserve the old
  import or command paths.
- `DQN/`, `SAC/`, `contextual_bandit/`, `paper_baseline/`, and
  `route_planning_agent/` contain lightweight shims so existing imports and
  saved-state lookups continue to resolve.
- `utils_sumo.py` is a deprecated facade. New logic must be added to a domain
  module instead of the facade.
- Shims may be removed only after external notebooks, research scripts, and
  serialized model compatibility have been audited.

## Verification layers

1. Characterization tests lock deterministic scheduling, charging queues,
   route XML operations, routing, and agent behavior.
2. API tests require old and new paths to reference the same Python objects.
3. Architecture tests enforce dependency boundaries and shim-only root files.
4. `compare_version_snapshot.py` runs the original and refactor in separate
   processes and compares semantic JSON snapshots.
5. `smoke_sumo_package.py` starts SUMO through TraCI and constructs live PyG
   graph features.

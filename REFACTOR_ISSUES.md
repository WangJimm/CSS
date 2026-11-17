# Refactoring issue log

This log records every issue found during the CSS reorganization and the
validation used for its resolution. All changes are confined to the refactor
copy; `D:\paper\CSS` remains unchanged.

## CSS-001 — Scenario-helper defaults did not match training assets

- Status: resolved (2026-08-23)
- Previous behavior: helper defaults used seeds 100–150 while the environment
  dataset and `DRL_train()` use 1–20.
- Resolution: `TRAINING_SEED_START=1` and `TRAINING_SEED_END=20` are now the
  single source for scheduling, random loading, and training defaults.
- Validation: all 40 seed/probability combinations resolve to existing general
  routes, EV routes, and matching power records.

## CSS-002 — Higher evaluation was the default without higher-test data

- Status: resolved (2026-08-23)
- Previous behavior: final evaluation defaulted to missing `higher_test_data`
  and `higher_test_data_power` directories.
- Resolution: standard testing data is now the default. Higher evaluation is
  explicit through `--higher-evaluation` and validates its directories before
  SUMO starts.
- Validation: parser tests lock the default and opt-in modes; missing-data tests
  require an actionable `FileNotFoundError`.

## CSS-003 — One testing power file had twice the EV cardinality

- Status: resolved (2026-08-23)
- Previous behavior: `power_ev_1800_p02_52.json` had 18 values for 9 EVs and
  the loader silently used the first 9.
- Resolution: retained those same first 9 values—the values historical runs
  actually consumed—and removed the unreachable surplus. All loaders now
  require exact EV/power cardinality.
- Validation: a complete training/testing/validation asset audit reports no
  mismatch; a synthetic malformed case must fail before SUMO is loaded.

## CSS-004 — Suspected missing route in `osm.sumocfg`

- Status: closed as not reproducible (2026-08-23)
- Finding: `env/osm.sumocfg` references
  `training_data\output_heavy_2400_101.rou.xml`; the file is present in the
  copied assets. The earlier report resulted from inspecting only the first
  lexicographically listed 1800-flow files.
- Validation: an XML asset test resolves the configured path, and a direct SUMO
  configuration smoke test loads it successfully.

## CSS-005 — Explicit scenario keys consumed global random values

- Status: resolved (2026-08-23)
- Previous behavior: `load_sumo()` sampled three random values and then
  overwrote them when `scenario_key` was supplied.
- Resolution: random sampling occurs only without an explicit key; callers may
  also inject a `random.Random` instance.
- Validation: a regression test proves the global RNG state is unchanged by an
  explicit scenario load.

## CSS-006 — Graph edge identifiers could be misaligned with tensors

- Status: resolved (2026-08-23)
- Previous behavior: tensors filtered edges touching excluded nodes while
  `edge_ids` retained all external edges.
- Resolution: tensors, edge IDs, adjacency, and length maps are built from one
  endpoint-filtered `graph_edges` list.
- Validation: a fake network containing one excluded endpoint requires equal
  tensor rows and IDs and excludes that edge from every mapping.

## CSS-007 — Empty recent-node history raised during graph initialization

- Status: resolved (2026-08-23)
- Previous behavior: `dynamic_node_feature_update()` unconditionally accessed
  `recent_node[-1]`.
- Resolution: empty history now leaves visit flags clear while decaying visit
  ages; target features remain populated.
- Validation: empty and populated histories are covered without changing the
  existing seven-column feature schema.

## CSS-008 — Lane resolver returned incompatible shapes

- Status: resolved (2026-08-23)
- Previous behavior: lane IDs returned one lane object and edge IDs returned a
  list; edge IDs containing underscores were also ambiguous.
- Resolution: `_safe_get_lanes()` tries an exact lane ID, falls back to an exact
  edge ID, and always returns a list. `_safe_get_lane` remains an alias for old
  callers. Multi-lane distance selection now keeps the best result across all
  source lanes.
- Validation: tests cover lane IDs, underscore-containing edge IDs, missing
  IDs, and the shortest result occurring before a longer lane result.

## CSS-009 — Traffic-light movement matching used ambiguous prefixes

- Status: resolved (2026-08-23)
- Previous behavior: `startswith()` allowed edge `edge1` to match lanes for
  `edge10`.
- Resolution: lane IDs are parsed with `lane2edge()` and compared exactly.
- Validation: a controlled-link test distinguishes `edge1/out` from
  `edge10/outbound`.

## CSS-010 — Remaining-distance checks accepted negative distances

- Status: resolved (2026-08-23)
- Previous behavior: several lane-length minus position calculations could
  return negative values, and any negative value satisfied the goal threshold.
- Resolution: remaining-distance helpers clamp ordinary overshoot to zero.
  Goal detection accepts only a small configurable numerical tolerance and
  rejects larger inconsistent positions.
- Validation: regression tests cover normal remainder, tolerated overshoot,
  large overshoot, and non-negative single-edge distance.

## CSS-011 — Action dimensions defaulted to a hard-coded 27

- Status: resolved (2026-08-23)
- Previous behavior: `action_mask_dqn()` and `oneHot()` could silently produce
  27-element tensors for another network.
- Resolution: action-mask size derives from `nid2idx` when omitted;
  `oneHot()` requires an explicit size. Both validate indices and dimensions
  with contextual errors.
- Validation: tests cover derived sizing, missing topology nodes, valid one-hot
  output, and out-of-range rejection. All active call sites pass network-derived
  dimensions.

## CSS-012 — Successful training exited with failure status

- Status: resolved (2026-08-23)
- Previous behavior: `sys.exit("end of training")` produced process status 1.
- Resolution: train-only completion prints the message and returns 0; module
  entry points raise `SystemExit(main())` so the return code is propagated.
- Validation: source/CLI tests reject `sys.exit()` calls inside the active
  implementation and exercise both module and compatibility entry points.

## CSS-013 — The eDQN route-planning checkpoint was unavailable

- Status: resolved (2026-08-23)
- Previous behavior: a missing `edqn_best_better_ratio.pt` emitted a warning
  and experiments could proceed with a randomly initialized route model.
- Investigation: the CSS tree did not contain the file, but a compatible route
  eDQN checkpoint was later identified at
  `D:\paper\RP\outputs\edqn_baseline\tmp_file.pt`. Its training marker records
  a completed 2500-episode `edqn_baseline` run, and the checkpoint records
  episode 2499 with agent type `edqn_refactored`.
- Resolution: the checkpoint was copied to the CSS route-model location. Its
  SHA-256 remained
  `98A7801EF35DD2797BEE39C1AD5A7CADC5274A29F154D72F3B145ADBCD8155AA`.
  CSS now uses the checkpoint's 231-element input schema: two 115-node one-hot
  vectors plus normalized simulation time. A compatibility module preserves
  the historical `DQN.edqn.EDQNShapeConfig` pickle path. Missing route models
  still fail before SUMO starts.
- Validation: the eval network, target network, and optimizer load strictly;
  episode 2499 is recovered; CPU inference returns four finite Q values. A
  SUMO/TraCI integration smoke test constructs `Env`, exposes `edqn` in the
  available route modes, selects it, and performs the same inference.

## CSS-014 — Reusable helpers terminated the entire Python process

- Status: resolved (2026-08-23)
- Previous behavior: configuration, departure handling, distance validation,
  and the legacy Dijkstra runner used `sys.exit()` for recoverable errors.
- Resolution: these paths now raise contextual `RuntimeError` exceptions. The
  caller can record, retry, or terminate deliberately without a package helper
  taking control of the process.
- Validation: a package-wide AST regression test rejects active `sys.exit()`
  calls, and a route-state mismatch test verifies the diagnostic exception.

## CSS-015 — Undefined traffic-light features failed silently

- Status: resolved (2026-08-23)
- Previous behavior: `tls_node_features()` contained only `pass`, so callers
  received `None` without learning that its feature schema was never defined.
- Resolution: the compatibility function raises `NotImplementedError` with
  directions to the supported graph-feature pipeline. It remains excluded from
  the public `css.graph` API.
- Validation: the legacy facade is tested to fail immediately with the expected
  explanation.

## CSS-016 — Incomplete route-policy rollouts crashed evaluation

- Status: resolved (2026-08-23)
- Previous behavior: when a route planner reached `MAX_DECISIONS_PER_EV`
  without arriving, its time fields remained `None`. Evaluation printed an
  incomplete-EV warning and immediately applied unary minus to `None`, masking
  the actual policy failure with a `TypeError`.
- Resolution: unfinished routes are now recorded as route-policy failures and
  receive the RP runner's conservative 3000-second (50-minute) censored travel
  cost. Charging-agent training receives that negative outcome instead of
  treating the failed route as cheap or aborting the entire run. Evaluation
  detail includes failure counts and per-EV reasons. Actual SUMO/TraCI failures
  remain fatal and produce contextual diagnostics.
- Validation: unit tests cover the decision-limit reason, the 50-minute floor,
  preservation of longer observed costs, unchanged completed EVs, and failure
  aggregation in evaluation summaries. A SUMO smoke run covers the originally
  failing eDQN/eDQN scenario.

## CSS-017 — The route eDQN checkpoint can reach the decision horizon

- Status: recorded; model-quality follow-up remains open (2026-08-23)
- Observation: in the standard `flow=1800`, seed 52, probability-index 2,
  min-waiting rollout, `EV_2` repeatedly selected route actions until reaching
  50 decisions without arriving at charging station `42496394`.
- Current handling: this is recorded as `reason=max_decisions` and receives the
  3000-second failure-horizon cost, so charging-agent training and evaluation
  remain statistically conservative and can continue.
- Recommended follow-up: retrain or fine-tune the route eDQN with explicit
  revisit/U-turn penalties and failure transitions, then compare success rate
  and penalized travel cost against the current checkpoint. Do not silently
  rescue failed eDQN routes with Dijkstra, because that would change the method
  being evaluated.

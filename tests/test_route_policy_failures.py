from __future__ import annotations

import types
import unittest

from css.experiment.config import EVState, MAX_DECISIONS_PER_EV, ROUTE_FAILURE_HORIZON_SEC
from css.experiment.training import (
    _append_case_metric,
    _settle_policy_failures,
    _summarize_method_metrics,
)


class RoutePolicyFailureTests(unittest.TestCase):
    def test_max_decisions_receives_fixed_horizon_cost(self):
        state = EVState(n_nodes=3)
        state.depart_time = 1000.0
        state.goal_cs = "cs-a"
        state.decision = MAX_DECISIONS_PER_EV

        failures = _settle_policy_failures({"EV_2": state}, now=1250.0)

        self.assertTrue(state.out)
        self.assertEqual(state.failure_reason, "max_decisions")
        self.assertEqual(state.travel_time, ROUTE_FAILURE_HORIZON_SEC)
        self.assertEqual(state.waiting_time, 0.0)
        self.assertEqual(state.charging_time, 0.0)
        self.assertEqual(failures[0]["observed_travel_sec"], 250.0)
        self.assertEqual(failures[0]["penalized_travel_sec"], ROUTE_FAILURE_HORIZON_SEC)

    def test_observed_cost_above_horizon_is_not_reduced(self):
        state = EVState(n_nodes=3)
        state.depart_time = 100.0
        state.decision = MAX_DECISIONS_PER_EV

        _settle_policy_failures({"EV_9": state}, now=4000.0)

        self.assertEqual(state.travel_time, 3900.0)

    def test_completed_state_is_unchanged(self):
        state = EVState(n_nodes=3)
        state.done = True
        state.travel_time = 120.0
        state.waiting_time = 30.0
        state.charging_time = 60.0

        failures = _settle_policy_failures({"EV_1": state}, now=999.0)

        self.assertEqual(failures, [])
        self.assertEqual((state.travel_time, state.waiting_time, state.charging_time), (120.0, 30.0, 60.0))

    def test_failure_count_is_preserved_in_evaluation_summary(self):
        store = {}
        cfg = types.SimpleNamespace(rd1=52, rd2=2, rd3=0)
        failure = {"ev": "EV_2", "reason": "max_decisions"}

        _append_case_metric(
            store,
            "1800",
            cfg,
            travel_cost_minutes=50.0,
            waiting_cost_minutes=0.0,
            route_failure_count=1,
            route_failures=[failure],
        )
        summary = _summarize_method_metrics(store)

        self.assertEqual(summary["overall"]["route_failure_count"], 1)
        self.assertEqual(summary["per_scenario"]["1800"]["route_failure_count"], 1)
        self.assertEqual(summary["overall"]["cases"][0]["route_failures"], [failure])


if __name__ == "__main__":
    unittest.main()

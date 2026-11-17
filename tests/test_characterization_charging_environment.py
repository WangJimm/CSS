from __future__ import annotations

import unittest

import numpy as np

from env import Env, normalize_state_minmax


class ChargingEnvironmentCharacterizationTests(unittest.TestCase):
    def make_environment(self) -> Env:
        environment = Env.__new__(Env)
        environment.cs_loc = ["cs-a", "cs-b", "cs-c"]
        environment.dis_table = {"origin": [0.1, 0.2, 0.3]}
        environment.eat = {"cs-a": 100.0, "cs-b": 700.0, "cs-c": 1_300.0}
        environment.usage = {
            "cs-a": [100.0, 200.0, 300.0],
            "cs-b": [700.0, 800.0, 900.0],
            "cs-c": [1_300.0, 1_400.0, 1_500.0],
        }
        environment.avai_port = {"cs-a": 0, "cs-b": 0, "cs-c": 0}
        return environment

    def test_state_uses_distance_power_and_relative_wait(self):
        environment = self.make_environment()

        state = environment.generate_sacd_state(700.0, "origin", 0.4)

        np.testing.assert_allclose(state, [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 1.0])

    def test_min_waiting_station_uses_stable_first_minimum(self):
        environment = self.make_environment()
        environment.eat.update({"cs-a": 500.0, "cs-b": 500.0, "cs-c": 900.0})

        self.assertEqual(environment.get_min_waiting_time_cs(400.0), "cs-a")

    def test_arrival_updates_the_earliest_available_charger(self):
        environment = self.make_environment()

        waiting, charging = environment.update(50.0, "cs-a", 0.2)

        self.assertEqual(waiting, 50.0)
        self.assertEqual(charging, 900.0)
        self.assertEqual(environment.usage["cs-a"], [1_000.0, 200.0, 300.0])
        self.assertEqual(environment.eat["cs-a"], 200.0)
        self.assertEqual(environment.avai_port["cs-a"], 1)

    def test_charge_and_minmax_formulas_are_stable(self):
        environment = self.make_environment()

        self.assertEqual(environment.charge(0.2), 900.0)
        self.assertEqual(normalize_state_minmax(5.0, 0.0, 10.0), 0.5)


if __name__ == "__main__":
    unittest.main()

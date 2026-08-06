import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from contextual_bandit import ContextualLinUCBAgent


class ContextualLinUCBTests(unittest.TestCase):
    def test_learns_reward_without_next_state(self):
        agent = ContextualLinUCBAgent(
            2,
            2,
            alpha=0.0,
            quadratic_features=False,
            enable_tensorboard=False,
            seed=7,
        )
        context = [0.25, 0.75]
        for _ in range(20):
            agent.memory.push(context, [0], [1.0], [999.0, 999.0], [False])
            agent.memory.push(context, [1], [-1.0], [-999.0, -999.0], [False])

        agent.update_parameters()

        self.assertEqual(agent.choose_action(context, eval=True), 0)
        self.assertEqual(agent.learn_step_counter, 40)
        np.testing.assert_array_equal(agent.action_counts, [20, 20])

    def test_context_uses_relative_queue_delay(self):
        env = SimpleNamespace(
            state_dim=7,
            action_dim=3,
            cs_loc=['a', 'b', 'c'],
            dis_table={'origin': [0.1, 0.2, 0.3]},
            eat={'a': 100.0, 'b': 700.0, 'c': 1_300.0},
        )
        agent = ContextualLinUCBAgent(env, enable_tensorboard=False)

        context = agent.build_state(700.0, 'origin', 0.4)

        np.testing.assert_allclose(context, [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 1.0])

    def test_choose_action_respects_available_actions(self):
        agent = ContextualLinUCBAgent(
            2,
            3,
            alpha=0.0,
            quadratic_features=False,
            enable_tensorboard=False,
            seed=7,
        )
        context = [0.25, 0.75]
        for _ in range(20):
            agent.memory.push(context, [0], [10.0], None, True)
            agent.memory.push(context, [1], [1.0], None, True)
            agent.memory.push(context, [2], [-1.0], None, True)
        agent.update_parameters()

        self.assertEqual(
            agent.choose_action(context, eval=True, available_actions=[1, 2]),
            1,
        )
        with self.assertRaises(ValueError):
            agent.choose_action(context, available_actions=[])

    def test_checkpoint_round_trip(self):
        agent = ContextualLinUCBAgent(2, 2, enable_tensorboard=False, seed=3)
        agent.memory.push([0.2, 0.8], [1], [2.0], None, True)
        agent.update_parameters()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'linucb.pt'
            agent.save_para(str(checkpoint))
            restored = ContextualLinUCBAgent(2, 2, enable_tensorboard=False, seed=3)
            restored.load_para(str(checkpoint))

        expected = agent.predict([0.2, 0.8])
        actual = restored.predict([0.2, 0.8])
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])


if __name__ == '__main__':
    unittest.main()

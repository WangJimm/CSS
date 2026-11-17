from __future__ import annotations

import unittest

import torch

from css.agents.routing.edqn import EDQNShapeConfig, route_eDQN
from css.charging import require_route_model


class EDQNCheckpointTests(unittest.TestCase):
    def test_rp_checkpoint_strictly_loads_and_runs_inference(self):
        shape = EDQNShapeConfig(num_nodes=115, max_neighbors=4)
        agent = route_eDQN(shape, device=torch.device("cpu"))

        episode = agent.load_full_model(
            filename=str(require_route_model("edqn")),
            strict=True,
            load_optimizer=True,
        )

        self.assertEqual(episode, 2499)
        self.assertEqual(agent.networks.eval_q_net.feature[0].in_features, 231)
        self.assertGreater(len(agent.optimizer.state), 0)

        base_state = torch.zeros(shape.num_nodes * 2, dtype=torch.float32)
        model_input = agent.concat_state_time(base_state, 1800.0).reshape(1, -1)
        q_values = agent.networks.eval_q_values(model_input)

        self.assertEqual(tuple(model_input.shape), (1, 231))
        self.assertEqual(tuple(q_values.shape), (1, 4))
        self.assertTrue(bool(torch.isfinite(q_values).all()))
        self.assertAlmostEqual(float(model_input[0, -1]), 0.5)


if __name__ == "__main__":
    unittest.main()

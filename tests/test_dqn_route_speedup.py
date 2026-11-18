"""DQN skips unused graph RPCs while retaining its live policy observation."""
import contextlib
from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch_geometric.data import Data

from css.agents.routing.dqn import DQNShapeConfig, route_DQN
from css.experiment import departures, planning
from css.experiment.config import EMA_feature, EVState
from css.graph.features import dynamic_node_feature_update


class DQNRouteSpeedupTests(unittest.TestCase):
    def fixture(self, agent):
        static = Data(x=torch.zeros(3, 3), edge_index=torch.tensor([[0], [1]]),
                      edge_attr=torch.ones(1, 3))
        env = SimpleNamespace(route_planning_agent=agent, nid2idx={'cur': 0, 'next': 1, 'goal': 2},
                              sumo_static_pack={'data_static': static}, net=Mock())
        env.net.getEdge.return_value.getToNode.return_value.getID.return_value = 'cur'
        st = EVState(3)
        st.goal_cs = 'goal'
        st.current_exit_edge = 'in'
        st.recent_nodes = deque([2, 1], maxlen=4)
        st.visit_age.fill_(.8)
        return env, st

    def test_first_and_followup_dqn_keep_full_observation_without_graph_queries(self):
        for method in (planning.first_plan, planning.second_plan):
            for schema in ('node_goal_tls_v0', 'node_goal_tls_time_v1'):
                with self.subTest(method=method.__name__, schema=schema):
                    agent = route_DQN(DQNShapeConfig(3, 2, observation_schema=schema),
                                      device=torch.device('cpu'), memory_capacity=1)
                    env, st = self.fixture(agent)
                    edge = Mock(); edge.getToNode.return_value.getID.return_value = 'next'
                    edge.getID.return_value = 'out'
                    fake = SimpleNamespace(vehicle=SimpleNamespace(getRoadID=Mock(return_value='in')),
                                           simulation=SimpleNamespace(getTime=Mock(return_value=1800.)))
                    emas = [EMA_feature(half_life_sec=5) for _ in range(3)]
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(planning, 'traci', fake))
                        stack.enter_context(patch.object(planning, 'candidate_next_junctions', return_value=(['next'], [edge])))
                        stack.enter_context(patch.object(planning, 'action_mask_dqn', return_value=torch.ones(3)))
                        stack.enter_context(patch.object(planning, 'find_tls_and_link_index_for_movement', return_value=('tls', 0)))
                        timing = stack.enter_context(patch.object(planning, 'time_to_green_and_remaining_green', return_value=(30., 60.)))
                        graph = stack.enter_context(patch.object(planning, 'build_data_runtime', side_effect=AssertionError('unused graph')))
                        features = stack.enter_context(patch.object(planning, 'build_candidate_action_features', side_effect=AssertionError('unused lanes')))
                        choose = stack.enter_context(patch.object(agent, 'choose_action_gnn', wraps=agent.choose_action_gnn))
                        apply = stack.enter_context(patch.object(planning, 'apply_next_junction', return_value='out'))
                        method('EV_7', st, env, *emas)
                    graph.assert_not_called(); features.assert_not_called()
                    timing.assert_called_once_with('tls', 0)
                    expected = torch.tensor([1., 0., 0., 0., 0., 1., .5, 1., 0., 0.] + ([.5] if schema.endswith('time_v1') else []))
                    torch.testing.assert_close(choose.call_args.args[0], expected, rtol=0, atol=0)
                    self.assertIsNone(choose.call_args.args[5])
                    self.assertIsNone(st.data_t)
                    self.assertEqual(list(st.recent_nodes), [2, 1, 0])
                    torch.testing.assert_close(st.visit_age, torch.tensor([[1.], [.76], [.76]]))
                    self.assertEqual(st.ever_visit.tolist(), [[1.], [0.], [0.]])
                    self.assertEqual(st.decision, 1)
                    self.assertEqual(st.routes, ['out'])
                    apply.assert_called_once()
                    self.assertTrue(all(not ema.val and not ema.last_t for ema in emas))

    def test_graph_consumers_and_unknown_policies_retain_original_graph_path(self):
        for agent in (SimpleNamespace(requires_route_graph=True, edge_feature_mode='static_occupancy'),
                      SimpleNamespace(use_gnn=False)):
            env, st = self.fixture(agent)
            graph = Data(x=torch.zeros(3, 3), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.ones(1, 8))
            expected, ever, age = dynamic_node_feature_update(graph, 2, deque([2, 1, 0]), st.ever_visit.clone(), st.visit_age.clone())
            action_features = torch.ones(1, 15)
            with patch.object(planning, 'build_data_runtime', return_value=graph) as build, \
                 patch.object(planning, 'build_candidate_action_features', return_value=action_features) as candidate:
                planning._prepare_route_graph(env, st, 'in', [], [], 0, 2, None, None, None)
            build.assert_called_once()
            candidate.assert_called_once()
            torch.testing.assert_close(st.data_t.x, expected.x, rtol=0, atol=0)
            torch.testing.assert_close(st.ever_visit, ever, rtol=0, atol=0)
            torch.testing.assert_close(st.visit_age, age, rtol=0, atol=0)
            self.assertIs(st.data_t.action_features, action_features)

    def test_non_graph_history_matches_graph_history_on_repeated_visits(self):
        env, st = self.fixture(SimpleNamespace(requires_route_graph=False))
        reference_recent = deque(st.recent_nodes, maxlen=4)
        ever, age = st.ever_visit.clone(), st.visit_age.clone()
        for source in (0, 1, 0, 2, 0):
            reference_recent.append(source)
            _, ever, age = dynamic_node_feature_update(env.sumo_static_pack['data_static'], 2, reference_recent, ever, age)
            with patch.object(planning, 'build_data_runtime') as build, patch.object(planning, 'build_candidate_action_features') as candidates:
                planning._prepare_route_graph(env, st, 'in', [], [], source, 2, None, None, None)
            build.assert_not_called(); candidates.assert_not_called()
            torch.testing.assert_close(st.ever_visit, ever, rtol=0, atol=0)
            torch.testing.assert_close(st.visit_age, age, rtol=0, atol=0)
            self.assertEqual(list(st.recent_nodes), list(reference_recent))

    def test_graph_station_selector_preserves_shared_ema_collection_with_dqn_route(self):
        env, st = self.fixture(SimpleNamespace(requires_route_graph=False))
        env.charging_uses_graph_state = True
        graph = Data(x=torch.zeros(3, 3), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.ones(1, 8))
        with patch.object(planning, 'build_data_runtime', return_value=graph) as collect, \
             patch.object(planning, 'build_candidate_action_features', return_value=torch.ones(1, 15)):
            planning._prepare_route_graph(env, st, 'in', [], [], 0, 2, None, None, None)
        collect.assert_called_once()
        self.assertIsNotNone(st.data_t)

    def test_departure_dispatch_refreshes_station_graph_requirement_when_agent_changes(self):
        env = SimpleNamespace()
        fake = SimpleNamespace(simulation=SimpleNamespace(getDepartedIDList=lambda: []))
        with patch.object(departures, 'traci', fake):
            for agent, expected in ((SimpleNamespace(uses_graph_state=True), True),
                                    (SimpleNamespace(), False), (None, False)):
                departures.compress_route_on_depart({}, env, agent)
                self.assertIs(env.charging_uses_graph_state, expected)


if __name__ == '__main__':
    unittest.main()

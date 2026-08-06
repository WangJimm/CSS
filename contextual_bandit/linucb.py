from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

def _scalar(value: Any) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError('Expected a scalar value')
    return float(array[0])


class BanditReplayBuffer:
    '''Store one-step feedback while accepting the transition API.'''

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
        self._pending = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state=None, done=None) -> None:
        del next_state, done
        context = np.asarray(state, dtype=np.float64).reshape(-1).copy()
        observation = (context, int(_scalar(action)), _scalar(reward))
        self.buffer.append(observation)
        self._pending.append(observation)

    def drain_pending(self):
        observations = list(self._pending)
        self._pending.clear()
        return observations

    def __len__(self) -> int:
        return len(self.buffer)


class ContextualLinUCBAgent:
    '''Disjoint LinUCB for single-step charging-station selection.'''

    uses_contextual_state = True

    def __init__(
        self,
        env_or_state_dim,
        n_actions: int | None = None,
        *,
        alpha: float = 1.0,
        ridge: float = 1.0,
        replay_capacity: int = 100_000,
        quadratic_features: bool = True,
        feature_clip: float = 10.0,
        seed: int | None = None,
        log_dir: str = 'runs/contextual_linucb',
        enable_tensorboard: bool = True,
    ) -> None:
        if hasattr(env_or_state_dim, 'state_dim') and hasattr(env_or_state_dim, 'action_dim'):
            self.env = env_or_state_dim
            self.state_dim = int(self.env.state_dim)
            self.n_actions = int(self.env.action_dim)
        else:
            self.env = None
            self.state_dim = int(env_or_state_dim)
            if n_actions is None:
                raise ValueError('n_actions is required when an environment is not provided')
            self.n_actions = int(n_actions)

        if self.state_dim <= 0 or self.n_actions <= 0:
            raise ValueError('state_dim and n_actions must be positive')
        if alpha < 0 or ridge <= 0 or feature_clip <= 0:
            raise ValueError('alpha must be non-negative; ridge and feature_clip must be positive')

        self.action_dim = self.n_actions
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.quadratic_features = bool(quadratic_features)
        self.feature_clip = float(feature_clip)
        self.feature_dim = 1 + self.state_dim * (2 if self.quadratic_features else 1)

        identity = np.eye(self.feature_dim, dtype=np.float64) / self.ridge
        self.a_inv = np.repeat(identity[None, :, :], self.n_actions, axis=0)
        self.b = np.zeros((self.n_actions, self.feature_dim), dtype=np.float64)
        self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.learn_step_counter = 0
        self.rng = np.random.default_rng(seed)

        self.batch_size = 1
        self.min_replay_size = 1
        self.memory = BanditReplayBuffer(replay_capacity)
        self.writer = _make_writer(log_dir) if enable_tensorboard else _NullWriter()

    def build_state(
        self,
        depart_time: float,
        depart_node,
        depart_power: float,
        travel_time=None,
        mean_speed=None,
        occupancy=None,
    ) -> list[float]:
        '''Build context with queue delay relative to the EV departure time.'''
        del travel_time, mean_speed, occupancy
        if self.env is None:
            raise RuntimeError('build_state requires the agent to be constructed with env')

        distances = list(self.env.dis_table[depart_node])
        waits = [
            max(float(self.env.eat[cs]) - float(depart_time), 0.0) / 600.0
            for cs in self.env.cs_loc
        ]
        context = distances + [float(depart_power)] + waits
        if len(context) != self.state_dim:
            raise ValueError(f'Expected context dimension {self.state_dim}, got {len(context)}')
        return context

    def _features(self, state) -> np.ndarray:
        context = np.asarray(state, dtype=np.float64).reshape(-1)
        if context.size != self.state_dim:
            raise ValueError(f'Expected context dimension {self.state_dim}, got {context.size}')
        context = np.nan_to_num(
            context,
            nan=0.0,
            posinf=self.feature_clip,
            neginf=-self.feature_clip,
        )
        context = np.clip(context, -self.feature_clip, self.feature_clip)
        parts = [np.ones(1, dtype=np.float64), context]
        if self.quadratic_features:
            parts.append(context * context)
        return np.concatenate(parts)

    def predict(self, state) -> tuple[np.ndarray, np.ndarray]:
        features = self._features(state)
        theta = np.einsum('aij,aj->ai', self.a_inv, self.b)
        means = theta @ features
        variances = np.einsum('i,aij,j->a', features, self.a_inv, features)
        uncertainty = np.sqrt(np.maximum(variances, 0.0))
        return means, uncertainty

    def choose_action(
        self,
        state,
        eval: bool = False,
        available_actions=None,
    ) -> int:
        means, uncertainty = self.predict(state)
        scores = means if eval else means + self.alpha * uncertainty
        if available_actions is None:
            eligible = np.arange(self.n_actions, dtype=np.int64)
        else:
            eligible = np.asarray(list(available_actions), dtype=np.int64).reshape(-1)
            if eligible.size == 0:
                raise ValueError('available_actions must contain at least one action')
            if np.any(eligible < 0) or np.any(eligible >= self.n_actions):
                raise ValueError(
                    f'available_actions must be within [0, {self.n_actions})'
                )
            eligible = np.unique(eligible)

        eligible_scores = scores[eligible]
        best_local = np.flatnonzero(
            np.isclose(
                eligible_scores,
                eligible_scores.max(),
                rtol=1e-10,
                atol=1e-12,
            )
        )
        return int(self.rng.choice(eligible[best_local]))

    def update_parameters(self, replay_buffer=None) -> float | None:
        buffer = replay_buffer if replay_buffer is not None else self.memory
        if not hasattr(buffer, 'drain_pending'):
            raise TypeError('ContextualLinUCBAgent requires BanditReplayBuffer')
        observations = buffer.drain_pending()
        if not observations:
            return None

        squared_errors = []
        for state, action, reward in observations:
            if not 0 <= action < self.n_actions:
                raise ValueError(f'Action {action} is outside [0, {self.n_actions})')
            features = self._features(state)
            theta = self.a_inv[action] @ self.b[action]
            error = reward - float(theta @ features)
            squared_errors.append(error * error)

            # Sherman-Morrison update of (ridge * I + X.T X)^-1.
            projected = self.a_inv[action] @ features
            denominator = 1.0 + float(features @ projected)
            self.a_inv[action] -= np.outer(projected, projected) / denominator
            self.b[action] += reward * features
            self.action_counts[action] += 1

        self.learn_step_counter += len(observations)
        mse = float(np.mean(squared_errors))
        self.writer.add_scalar('loss/reward_prediction_mse', mse, self.learn_step_counter)
        self.writer.add_scalar('stats/observations', self.learn_step_counter, self.learn_step_counter)
        for action, count in enumerate(self.action_counts):
            self.writer.add_scalar(f'stats/action_{action}_count', int(count), self.learn_step_counter)
        return mse

    def save_para(self, filename: str = 'contextual_linucb.pt') -> None:
        path = Path(filename)
        if path.parent != Path('.'):
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as output:
            np.savez_compressed(
                output,
                agent_type=np.asarray('contextual_linucb'),
                state_dim=np.asarray(self.state_dim),
                n_actions=np.asarray(self.n_actions),
                alpha=np.asarray(self.alpha),
                ridge=np.asarray(self.ridge),
                quadratic_features=np.asarray(self.quadratic_features),
                feature_clip=np.asarray(self.feature_clip),
                a_inv=self.a_inv,
                b=self.b,
                action_counts=self.action_counts,
                learn_step_counter=np.asarray(self.learn_step_counter),
            )
        print(f'contextual LinUCB parameters saved to {path}')

    def load_para(self, filename: str = 'contextual_linucb.pt', strict: bool = True) -> None:
        with np.load(filename, allow_pickle=False) as checkpoint:
            loaded_a = checkpoint['a_inv'].astype(np.float64, copy=True)
            loaded_b = checkpoint['b'].astype(np.float64, copy=True)
            loaded_counts = checkpoint['action_counts'].astype(np.int64, copy=True)
            loaded_steps = int(checkpoint['learn_step_counter'])

        expected_a = (self.n_actions, self.feature_dim, self.feature_dim)
        expected_b = (self.n_actions, self.feature_dim)
        if loaded_a.shape != expected_a or loaded_b.shape != expected_b:
            message = (
                f'Checkpoint shape mismatch: A_inv={loaded_a.shape}, b={loaded_b.shape}, '
                f'expected {expected_a} and {expected_b}'
            )
            if strict:
                raise ValueError(message)
            print(f'[warn] {message}')
            return

        self.a_inv = loaded_a
        self.b = loaded_b
        self.action_counts = loaded_counts
        self.learn_step_counter = loaded_steps
        print(f'contextual LinUCB parameters loaded from {filename}')


class _NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        del args, kwargs

    def close(self) -> None:
        return None


def _make_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print('[warn] TensorBoard is unavailable; LinUCB metrics will not be written')
        return _NullWriter()
    return SummaryWriter(log_dir=log_dir)

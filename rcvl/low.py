import os
import pickle
import numpy as np

from gym_simple_minigrid.minigrid import SimpleMiniGridEnv

from .rl_algs.ddqn import DDQNStateGoal
from .utils.utils import ReplayBuffer, HERTransitionCreator

# https://github.com/rafelps/learning-recursive-goal-proposal
# Note: Not the author's implementation


class LowPolicy:
    def __init__(self, env: SimpleMiniGridEnv, gamma: float = 1., tau: float = 0.005, br_size: int = 5e5):
        self.env = env
        state_shape = env.observation_space.shape[0]
        action_dims = env.action_space.n
        goal_shape = state_shape
        # Init DDQN algorithm, base learner for low agent
        self.alg = DDQNStateGoal(state_dim=state_shape, action_dim=action_dims, goal_dim=goal_shape,
                                 gamma=gamma, tau=tau, hidden_dims=(256, 256))

        # Init ER and HER
        self.replay_buffer = ReplayBuffer(br_size)
        self.her = HERTransitionCreator(env.state_goal_mapper)

        # Reachable and Allowed buffers
        self.run_steps = list()
        self.reachable_buffer = set()
        self.allowed_buffer = set()

    def select_action(self, state: np.ndarray, goal: np.ndarray, epsilon: float):
        # Apply epsilon-greedy exploration strategy
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.alg.select_action(state, goal)
        return action

    def update(self, n_updates: int, batch_size: int):
        if len(self.replay_buffer) > 0:
            for _ in range(n_updates):
                self.alg.update(self.replay_buffer, batch_size)

    def add_transition(self, transition: tuple):
        self.replay_buffer.add(*transition)
        self.her.add(*transition)

    def on_episode_end(self):
        # Compute 'future' her and insert hindsight transitions in buffer
        self.her.create_and_insert(self.replay_buffer)

    def add_run_step(self, state: np.ndarray):
        self.run_steps.append(state)

    def create_reachable_transitions(self, goal: np.ndarray, achieved: bool, low_h: int):
        # Store reachable transition: (state, goal)
        for i, s_1 in enumerate(self.run_steps):
            for s_2 in self.run_steps[i + 1:]:
                self.reachable_buffer.add(tuple(s_1) + tuple(s_2))
        if not achieved:
            pass
        self.run_steps = list()

    def is_reachable(self, state: np.ndarray, goal: np.ndarray, epsilon=0.):
        if np.random.random() < epsilon:
            return True
        else:
            return (tuple(state) + tuple(goal)) in self.reachable_buffer

    def save(self, path: str):
        self.alg.save(path, "low")
        with open(os.path.join(path, "low_reachable.pkl"), 'wb') as f:
            pickle.dump(self.reachable_buffer, f)

    def load(self, path: str):
        self.alg.load(path, "low")
        with open(os.path.join(path, "low_reachable.pkl"), 'rb') as f:
            self.reachable_buffer = pickle.load(f)


import numpy as np
from gym_simple_minigrid.minigrid import SimpleMiniGridEnv
import torch
from .rl_algs.sac import SACStateGoal
from .utils.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HighPolicy:
    def __init__(self, env: SimpleMiniGridEnv, gamma: float = 1., tau: float = 0.005, br_size: int = 1e6,
                 lr_critic: float = 3e-4):
        self.env = env

        state_shape = env.observation_space.shape[0]
        goal_shape = state_shape
        action_shape = goal_shape

        # For clapping actions
        action_low = env.observation_space.low
        action_high = env.observation_space.high
        action_high_corrected = 1 + action_high  # To adapt discrete env into SAC (continuous actions)

        action_bound = (action_high_corrected - action_low) / 2
        action_offset = (action_high_corrected + action_low) / 2

        # Init SAC algorithm, base learner for high agent
        self.alg = SACStateGoal(state_shape, action_shape, goal_shape, action_bound, action_offset, gamma, tau,
                                critic_lr=lr_critic)

        self.clip_low = np.concatenate((action_low, np.array([0])))
        self.clip_high = np.concatenate((action_high, np.array([3])))

        self.clip_low = action_low
        self.clip_high = action_high

        self.replay_buffer = ReplayBuffer(br_size)
        self.alg.goal_list = [set() for _ in range(self.env.height * self.env.width)]

        self.episode_runs = list()

    def select_action(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # Get the index of goal in goal\_list
        current_1d_goal = self.env.location_to_number(goal)
        # Subgoals suggestion potentially all of the states in the vicinity.
        list_possible_actions = list(self.alg.goal_list[current_1d_goal])
        if bool(list_possible_actions):
            # Broadcasting state and goal to vicinity dimensions so it is possible to use batch calc of Q_val
            state_list = [state for _ in range(len(list_possible_actions))]
            goal_list = [goal for _ in range(len(list_possible_actions))]
            q_values = self.calc_q_vals(state_list, list_possible_actions, goal_list)
            # Select the action that results with highest Q-val estimation
            max_idx = np.argmax(np.array(q_values))
            return list_possible_actions[max_idx]
        else:  # If the vicinity of current goal is empty - use the explicit representation
            action = self.alg.select_action(state, goal, False)
            action = np.floor(action)  # Discretization of SAC continuous output
            action = np.clip(action, self.clip_low, self.clip_high)
            return action.astype(np.int)

    def select_action_test(self, state: np.ndarray, goal: np.ndarray, add_noise: bool = False) -> np.ndarray:
        action = self.select_action(state, goal)
        return action

    def add_run_info(self, info: tuple):
        self.episode_runs.append(info)

    def add_penalization(self, transition: tuple):
        self.replay_buffer.add(*transition)

    def solution_to_vicinity(self, solution, radius, symmetry):
        # Here we rollout the high level steps and store states in the relevant goal vicinity
        solution.reverse()
        for i, element in enumerate(solution):
            goal_1dim = self.env.location_to_number(element)
            for j in range(1, len(solution) - i):
                if self.env.state_goal_mapper(element) != self.env.state_goal_mapper(solution[i + j]):
                    self.alg.goal_list[goal_1dim].add(solution[i + j])
                    if symmetry:  # Store also symmetric relation
                        curr_state_1dim = self.env.location_to_number(solution[i + j])
                        self.alg.goal_list[curr_state_1dim].add(element)
                if j >= radius:  # Stop storing when reaching the limit of the radius
                    break

    def on_episode_end(self):
        # Transitions creation based on MonteCarlo-inspired approach
        # Hindsight action relabelling: the next state as proposed subgoal (as if low level acts optimally)
        for i, (state_1, next_state_1) in enumerate(self.episode_runs):
            for j, (_, next_state_3) in enumerate(self.episode_runs[i + 1:], i + 1):
                for k, (_, next_state_2) in enumerate(self.episode_runs[i:j], i):
                    self.replay_buffer.add(state_1,  # state
                                           next_state_2,  # action (proposed goal)
                                           -(j - i + 1),  # reward (# runs)
                                           next_state_1,  # next_state
                                           next_state_3,  # goal
                                           True)
        self.episode_runs = list()

    def calc_q_vals(self, state, action, goal):
        # Function for forward a batch of (state,action,goal) to Q_val netowrk
        state_tensor = torch.FloatTensor(state).to(device)
        action_tensor = torch.FloatTensor(action).to(device)
        goal_tensor = torch.FloatTensor(goal).to(device)
        state_goal = torch.cat([state_tensor, goal_tensor], dim=-1)
        with torch.no_grad():
            q_val = self.alg.q_2(state_goal, action_tensor).squeeze(-1).cpu().numpy()
        return q_val

    def calc_v_vals(self, state, goal):
        # Function for forward a batch of (state,goal) to V_val netowrk
        state_tensor = torch.FloatTensor(state).to(device)
        goal_tensor = torch.FloatTensor(goal).to(device)
        state_goal = torch.cat([state_tensor, goal_tensor], dim=-1)
        with torch.no_grad():
            v_val = self.alg.value(state_goal).cpu().numpy()
        return v_val

    def update(self, n_updates: int, batch_size: int):
        if len(self.replay_buffer) > 0:
            for _ in range(n_updates):
                self.alg.update(self.replay_buffer, batch_size)

    def calc_mean_goal_list_len(self):
        # For calculation of goal_list mean side in terms of number of states accumulated.
        len_all = [len(self.alg.goal_list[i]) for i in range(len(self.alg.goal_list)) if
                   len(self.alg.goal_list[i]) != 0]
        return np.mean(len_all)

    def save(self, path: str):
        self.alg.save(path, "high")

    def load(self, path: str):
        self.alg.load(path, "high")

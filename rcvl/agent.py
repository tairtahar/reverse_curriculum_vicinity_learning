import os
import numpy as np

from gym_simple_minigrid.minigrid import SimpleMiniGridEnv
from typing import Callable, Tuple
import pickle
from .high import HighPolicy
from .low import LowPolicy
import time


class Sample_goal:
    def __init__(self, cur_env: SimpleMiniGridEnv, lr_critic=3e-4):
        self.env = cur_env
        self.low = LowPolicy(cur_env)
        self.high = HighPolicy(cur_env, lr_critic=lr_critic)
        self.logs = list()
        self.radius = 1
        self.back_forth = 5

    def train(self, n_samples_low: int, n_episodes: int, low_h: int, high_h: int, test_each: int, n_episodes_test: int,
              update_each: int, n_updates: int, batch_size: int, epsilon_range: Callable, **kwargs):
        acquisition_time, len_list = self.preliminary_acquisition(n_samples_low, low_h, update_each,
                                                                  batch_size, epsilon_range, kwargs['symmetry'],
                                                                  kwargs['n_targets'], kwargs['flatten'])
        self.radius = kwargs['radius_h']
        self.back_forth = kwargs['back_forth_low']
        print("Acquisition time lasted " + str(acquisition_time / 60) + " minutes")
        mean_len = self.high.calc_mean_goal_list_len()
        self.save_low(os.path.join('logs', kwargs['job_name']), len_list)
        print("mean goal list length is " + str(mean_len))
        start_time = time.time()
        for episode in range(n_episodes):
            epsilon = epsilon_range(episode)

            # Init episode variables
            subgoals_count = 0
            max_env_steps = False

            # Environment initialization
            state, episode_goal = self.env.reset()
            episode_goal = np.concatenate((episode_goal, np.random.randint(0, 3, 1)))  # Add 3rd dim to the goal
            goal_stack = [episode_goal]
            solution = [tuple(state)]

            while True:
                goal = goal_stack[-1]

                # Check if reachable
                reachable = self.low.is_reachable(state, goal, epsilon)

                if not reachable:  # high level gets into action
                    subgoals_count += 1
                    # Limit for number of subgoals
                    if subgoals_count > high_h:
                        break  # Exeeding proposals limit. Finish episode.

                    # New subgoal proposal
                    new_goal = self.high.select_action(state, goal)
                    # Penalty for proposal of the same location as current state or current goal
                    if np.array_equal(self.env.state_goal_mapper(new_goal), self.env.state_goal_mapper(goal)) or \
                            np.array_equal(self.env.state_goal_mapper(new_goal), self.env.state_goal_mapper(state)):
                        self.high.add_penalization((state, new_goal, -high_h, state, goal, True))  # ns not used
                    else:
                        goal_stack.append(new_goal)

                else:  # Low level gets into action.
                    # If we got here, the mini trajectory is familiar to the low level (Reachable) or we are exploring.
                    # The low policy is getting into action

                    state_high = state

                    # Initialize run variables
                    achieved = self._goal_achived(state, goal)
                    low_fwd = 0
                    low_actions = 0

                    self.low.add_run_step(state)

                    # Low policy (low_h) steps of mini trajectory
                    while low_fwd < low_h and low_actions < 2 * low_h and not achieved:
                        action = self.low.select_action(state, goal, epsilon)
                        next_state, reward, done, info = self.env.step(action)
                        achieved = self._goal_achived(next_state, goal)
                        self.low.add_transition((state, action, int(achieved) - 1, next_state, goal, achieved))
                        state = next_state
                        self.low.add_run_step(state)

                        if action == SimpleMiniGridEnv.Actions.forward:
                            low_fwd += 1  # count only forward passes
                        low_actions += 1

                        # Max env steps
                        if done and len(info) > 0:
                            max_env_steps = True
                            break

                    next_state_high = state  # Final state of mini-trajectory
                    solution.append(tuple(state))

                    # Create reachable transitions from run info
                    self.low.create_reachable_transitions(goal, achieved, low_h)

                    # Add run info for high agent to create transitions
                    if not np.array_equal(state_high, next_state_high):
                        self.high.add_run_info((state_high, next_state_high))

                    # Update goal stack
                    while len(goal_stack) > 0 and self._goal_achived(next_state_high, goal_stack[-1]):
                        goal_stack.pop()

                    # Zero steps means the low policy is stuck. We force it to move.
                    assert low_actions != 0

                    if len(goal_stack) == 0:  # Successfully reached the ultimate goal. End the episode
                        break

                    elif max_env_steps:  # Episode ends unsuccessfully since environment limit is reached
                        break

            # Convert the path to neighbors in goal_list + transitions + HER
            self.high.solution_to_vicinity(solution, self.radius, kwargs['symmetry'])
            self.high.on_episode_end()
            self.low.on_episode_end()

            # Update networks
            if (episode + 1) % update_each == 0:
                self.high.update(n_updates, batch_size)
                self.low.update(n_updates, batch_size)

            # Test to validate training
            if (episode + 1) % test_each == 0:
                subg, subg_a, steps, steps_a, max_subg, sr, low_sr = self.test(n_episodes_test, low_h, high_h)
                print(f"Episode {episode + 1:5d}: {100 * sr:5.1f}% Achieved")
                curr_time = (time.time() - start_time) / 60
                self.logs.append([episode, subg, subg_a, steps, steps_a, max_subg, sr, low_sr,
                                  len(self.high.replay_buffer), len(self.low.replay_buffer),
                                  len(self.low.reachable_buffer), len(self.low.allowed_buffer),
                                  self.high.calc_mean_goal_list_len(), curr_time])
                self.save(os.path.join('logs', kwargs['job_name']))

    def preliminary_acquisition(self, n_samples: int, low_h: int, update_each: int, batch_size: int,
                                epsilon_range: Callable, symmetry, n_targets, flatten):
        low_train_start = time.time()
        mean_len_list = []
        for sample in range(n_samples):
            epsilon = epsilon_range(sample)
            state, episode_goal = self.env.reset()
            for target in range(n_targets):
                goal = np.concatenate((episode_goal, np.random.randint(0, 3, 1)))
                solution = [tuple(state)]
                achieved = self._goal_achived(state, goal)
                if not achieved:
                    for run_iter in range(self.back_forth):
                        last_state, achieved = self.run_steps(state, goal, low_h, epsilon)
                        self.low.create_reachable_transitions(goal, achieved, low_h)
                        if not flatten:
                            goal = state
                        state = last_state
                        solution.append(tuple(last_state))
                        self.low.on_episode_end()

                self.high.solution_to_vicinity(solution, self.radius, symmetry)

            # Update networks
            if (sample + 1) % update_each == 0:
                self.low.update(3, batch_size)

            if (sample + 1) % 50 == 0:
                print("low sampling target " + str(sample + 1))
                mean_len_list.append(self.high.calc_mean_goal_list_len())
        low_train_end = time.time()
        low_time = low_train_end - low_train_start
        return low_time, mean_len_list

    def _goal_achived(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return np.array_equal(state, goal)

    def run_steps(self, state: np.ndarray, goal: np.ndarray, low_h: int, epsilon: float):
        low_actions = low_fwd = 0
        self.low.add_run_step(state)
        achieved = self._goal_achived(state, goal)
        while low_fwd < low_h and low_actions < 2 * low_h and not achieved:
            action = self.low.select_action(state, goal, epsilon)
            next_state, reward, done, info = self.env.step(action)
            # Check if last subgoal is achieved (not episode's goal)
            achieved = self._goal_achived(next_state, goal)
            self.low.add_transition(
                (state, action, int(achieved) - 1, next_state, goal, achieved))

            state = next_state
            self.low.add_run_step(state)

            if action == SimpleMiniGridEnv.Actions.forward:
                low_fwd += 1
            low_actions += 1

        return state, achieved

    def test(self, n_episodes: int, low_h: int, high_h: int, render: bool = False, **kwargs) -> Tuple[np.ndarray, ...]:
        if render:
            return self._test_render(n_episodes, low_h, high_h)
        else:
            return self._test(n_episodes, low_h, high_h)

    def _test(self, n_episodes: int, low_h: int, high_h: int) -> Tuple[np.ndarray, ...]:
        # Log metrics
        log_proposals = list()
        log_proposals_a = list()
        log_steps = list()
        log_steps_a = list()
        log_success = list()
        log_low_success = list()
        log_max_proposals = list()

        for episode in range(n_episodes):
            # Init episode variables
            subgoals_count = 0
            low_steps_ep = 0
            max_env_steps = max_subgoals_proposed = low_stuck = False

            state, episode_goal = self.env.reset()  # Initialize environment
            episode_goal = np.concatenate((episode_goal, np.random.randint(0, 3, 1)))  # add 3rd dim to goal
            goal_stack = [episode_goal]

            while True:
                goal = goal_stack[-1]

                # Check if reachable
                reachable = self.low.is_reachable(state, goal, 0)

                if not reachable:
                    subgoals_count += 1
                    # Limit for number of subgoals
                    if subgoals_count > high_h:
                        max_subgoals_proposed = True
                        break  # Exeeding proposals limit. Finish episode.

                    # New subgoal proposal
                    new_goal = self.high.select_action_test(state, goal)
                    goal_stack.append(new_goal)

                else:
                    state_high = state

                    # Initialize run variables
                    achieved = self._goal_achived(state, goal)
                    low_fwd = 0
                    low_actions = 0

                    # Low policy (low_h) steps of mini trajectory
                    while low_fwd < low_h and low_actions < 2 * low_h and not achieved:
                        action = self.low.select_action(state, goal, 0)
                        next_state, reward, done, info = self.env.step(action)
                        achieved = self._goal_achived(next_state, goal)
                        state = next_state
                        if action == SimpleMiniGridEnv.Actions.forward:
                            low_fwd += 1  # count only forward passes
                        low_actions += 1
                        low_steps_ep += 1

                        # Max env steps
                        if done and len(info) > 0:
                            max_env_steps = True
                            break

                    next_state_high = state  # Final state of mini-trajectory
                    log_low_success.append(achieved)

                    # Pop out if reached to goals/subgoals
                    while len(goal_stack) > 0 and self._goal_achived(next_state_high, goal_stack[-1]):
                        goal_stack.pop()

                    if len(goal_stack) == 0:  # Successfully finishing episode
                        break

                    elif np.array_equal(state_high, next_state_high):
                        low_stuck = True
                        break

                    elif max_env_steps:  # Episode ends unsuccessfully since environment limit is reached
                        break

            # Log metrics
            episode_achieved = not max_subgoals_proposed and not max_env_steps and not low_stuck
            log_success.append(episode_achieved)
            log_max_proposals.append(max_subgoals_proposed)
            log_proposals.append(min(subgoals_count, high_h))
            log_steps.append(low_steps_ep)
            if episode_achieved:
                log_proposals_a.append(min(subgoals_count, high_h))
                log_steps_a.append(low_steps_ep)

        if len(log_proposals_a) == 0:
            log_proposals_a = [0]
            log_steps_a = [0]

        return np.array(log_proposals).mean(), np.array(log_proposals_a).mean(), np.array(log_steps).mean(), \
               np.array(log_steps_a).mean(), np.array(log_max_proposals).mean(), np.array(log_success).mean(), \
               np.array(log_low_success).mean()

    def _test_render(self, n_episodes: int, low_h: int, high_h: int) -> Tuple[np.ndarray, ...]:

        # Log metrics
        log_proposals = list()
        log_proposals_a = list()
        log_steps = list()
        log_steps_a = list()
        log_success = list()
        log_low_success = list()
        log_max_proposals = list()

        for episode in range(n_episodes):
            # Init episode variables
            subgoals_count = 0
            low_steps_ep = 0
            max_env_steps = max_subgoals_proposed = low_stuck = False

            # Environment initialization
            state, episode_goal = self.env.reset()
            self.env.render()
            episode_goal = np.concatenate((episode_goal, np.random.randint(0, 3, 1)))
            goal_stack = [episode_goal]

            while True:
                goal = goal_stack[-1]

                # Check if reachable
                reachable = self.low.is_reachable(state, goal, 0)

                if not reachable:  # high level gets into action
                    subgoals_count += 1
                    if subgoals_count > high_h:
                        max_subgoals_proposed = True
                        break   # Exeeding proposals limit. Finish episode.

                    # New subgoal proposal
                    new_goal = self.high.select_action_test(state, goal)
                    goal_stack.append(new_goal)
                    self.env.add_goal(self.env.state_goal_mapper(new_goal))
                    self.env.render()

                else:
                    state_high = state

                    # Initialize run variables
                    achieved = self._goal_achived(state, goal)
                    low_fwd = 0
                    low_actions = 0

                    # Low policy (low_h) steps: mini-trajectory.
                    while low_fwd < low_h and low_actions < 2 * low_h and not achieved:
                        action = self.low.select_action(state, goal, 0)
                        next_state, reward, done, info = self.env.step(action)
                        self.env.render()
                        achieved = self._goal_achived(next_state, goal)
                        state = next_state

                        if action == SimpleMiniGridEnv.Actions.forward:
                            low_fwd += 1
                        low_actions += 1
                        low_steps_ep += 1

                        if done and len(info) > 0:
                            max_env_steps = True
                            break

                    next_state_high = state

                    log_low_success.append(achieved)

                    # Pop out for goal_stack if goals were achieved
                    while len(goal_stack) > 0 and self._goal_achived(next_state_high, goal_stack[-1]):
                        goal_stack.pop()
                        self.env.remove_goal()
                    self.env.render()

                    if len(goal_stack) == 0:  # Successfully finished episode
                        break

                    elif np.array_equal(state_high, next_state_high):  # unsuccessful finished
                        low_stuck = True
                        break

                    # Check episode completed due to Max Env Steps
                    elif max_env_steps:
                        break

            # Log metrics
            episode_achieved = not max_subgoals_proposed and not max_env_steps and not low_stuck
            log_success.append(episode_achieved)
            log_max_proposals.append(max_subgoals_proposed)
            log_proposals.append(min(subgoals_count, high_h))
            log_steps.append(low_steps_ep)
            if episode_achieved:
                log_proposals_a.append(min(subgoals_count, high_h))
                log_steps_a.append(low_steps_ep)

        # Avoid taking the mean of an empty array
        if len(log_proposals_a) == 0:
            log_proposals_a = [0]
            log_steps_a = [0]

        return np.array(log_proposals).mean(), np.array(log_proposals_a).mean(), np.array(log_steps).mean(), \
               np.array(log_steps_a).mean(), np.array(log_max_proposals).mean(), np.array(log_success).mean(), \
               np.array(log_low_success).mean()

    def save_low(self, path, len_list):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, f"len_goal_list.pickle"), 'wb') as f:
            # for _set in self.goal_list:
            pickle.dump(len_list, f)

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.high.save(path)
        self.low.save(path)
        with open(os.path.join(path, f"logs.npy"), 'wb') as f:
            np.save(f, np.array(self.logs))

    def load(self, path: str):
        self.high.load(path)
        self.low.load(path)

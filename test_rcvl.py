import os
import argparse
import gym
import gym_simple_minigrid  # just to register envs
import numpy as np
from rcvl.agent import Sample_goal

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_name', type=str, default='rcvl_four_rooms', help='Name of the checkpoint '
                                                                                   'subdirectory')
parser.add_argument('--env', type=str, default='Simple-MiniGrid-FourRooms-15x15-v0', help='Environment to use')
parser.add_argument('--seed', type=int, default=12345, help='Seed to control the testing random generations')
parser.add_argument('--n_episodes', type=int, default=1000, help='Number of episodes')
parser.add_argument('--low_h', type=int, default=4, help='Low horizon: maximum number of steps the low agent can do '
                                                         'every time it is used')
parser.add_argument('--high_h', type=int, default=15, help='High horizon: maximum number of subgoal proposals per '
                                                           'episode')
parser.add_argument('--render', action='store_true', help='Visualize the agent\'s learned policy')


args = parser.parse_args()
args = vars(args)

# Make env
env = gym.make(args['env'])

# Seed everything
env.seed(args['seed'])
np.random.seed(args['seed'])

# Test
print(f"Testing checkpoint {args['checkpoint_name']}...")
tester = Sample_goal(env)

tester.load(os.path.join('checkpoints', args['checkpoint_name']))


_, subg_a, _, steps_a, max_subg, sr, low_sr = tester.test(**args)

print(f"Evaluation after {args['n_episodes']} testing episodes:")
print(f"\t{sr*100:.1f}% episodes achieved")
print(f"\t{max_subg*100:.1f}% failed due to max subgoal proposals")
print(f"\t Low policy success percentage: {low_sr*100:.1f}% ")
print("Among the completed episodes:")
print(f"\tAverage number of subgoals proposed per episode = {subg_a:.2f}")
print(f"\tAverage number of steps performed per episode = {steps_a:.2f}")

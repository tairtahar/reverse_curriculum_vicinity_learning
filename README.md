![python](https://img.shields.io/badge/Python-3.7-orange) ![torch](https://img.shields.io/badge/pytorch-1.7.1-brightgreen) ![gym](https://img.shields.io/badge/gym-0.19-red) 
# RCVL: Reverse Curriculum Hierarchical Vicinity Learning.

Code for my final master thesis: Reverse Curriculum Hierarchical Recursive Learning.

This repository was developed for master thesis in Artificial Intelligence. 
It contains an hierarchical algorithms based on two levels in which the higher draws the high level 
path with milestones (subgoals/starting points) and the low hierarchy performs the primitive steps between those 
milestones. 

The algorithm is trained and tested in Simple Minigrid environemt: Empty Room 15x15 and FourRooms 15x15. 


## Dependencies

- [Python][python] 3.7
- [PyTorch][pytorch] 1.7.1
- [NumPy][numpy] 1.19.2
- [OpenAI Gym][gym] 0.17.2
- [Gym Simple MiniGrid][smg] 2.0.0 (https://github.com/tairtahar/gym-simple-minigrid.git)

## Usage

Running experiments:
1. For execution of training please run  `train_rcvl.py`.
Make sure to insert job_name and to adjust any of the parameters if needed.
2. For execution of testing please run `test_rcvl.py`. 
Make sure to insert checkpoint_name, to add the relevant files into the checkpoints directory, and to adjust any of the parameters if needed. 
The files default checkpoint_name is the algorithms that were presented in the thesis for four_rooms environment. 
In addition, in the checkpoint directory, it is possible to find the checkpoint for empty room. 

~


For rendering the test, add `--render` to the configuration file. 



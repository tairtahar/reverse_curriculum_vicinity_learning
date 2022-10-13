# Reverse Curriculum Hierarchical Recursive Learning: RCVL and RCRL

Code for my final master thesis: Reverse Curriculum Hierarchical Recursive Learning.

This master thesis presents two hierarchical algorithms based on two levels in which the higher draws the high level 
path with milestones (subgoals/starting points) and the low hierarchy performs the primitive steps between those 
milestones. The algorithms are as following:
1. RCRL - Reverse Curriculum Recursive Learning.
2. RCVL - Reverse Curriculum Vicinity Learning.

They are trained and tested in Simple Minigrid environemt: Empty Room 15x15 and FourRooms 15x15. 

To execute the project please refer to the dependecies bellow. Then please update the SimpleMinigrid with the adjusted 
version in the attached folder: gym-simple-minigrid. That will allow the newly added features of the environment 
available for use. 

## Dependencies

This project has been developed using:

- [Python][python] 3.7
- [PyTorch][pytorch] 1.7.1
- [NumPy][numpy] 1.19.2
- [OpenAI Gym][gym] 0.17.2
- [Gym Simple MiniGrid][smg] 2.0.0 (https://github.com/tairtahar/gym-simple-minigrid.git)

## Usage
Each of the algorithm implementations, with the relevant files is located in the corresponding directory:
1. reverse_curriculum_recursive_learning.
2. reverse_curriculum_vicinity_learning.

Running experiments:
1. For execution of training of each of them please run  `train_xxxx.py` being `xxxx` the name of the algorithm.
Make sure to insert job_name and to adjust any of the parameters if needed.
2. For execution of testing of each of them please run `test_xxx.py` being `xxxx` the name of the algorithm. 
Make sure to insert checkpoint_name and to adjust any of the parameters if needed. 
The files default checkpoint_name is the algorithms that were presented in the thesis for four_rooms environment. 
In addition, in the checkpoint directory, it is possible to find the checkpoint for empty room. 

~

Refer to each of the files  `train_xxxx.py` and `test_xxx.py` parameters explanation by:
```
$ python train_xxxx.py -h
```
or
```
$ python test_xxxx.py -h
```
For rendering the test, add `--render` to the configuration file. 



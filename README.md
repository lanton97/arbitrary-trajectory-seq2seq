# manifold-convoy

## Documentation

The image below outlines the control loop for the system we are considering. The blue background rectangle highlights the components we are interested in in this context. The NN code will act as a trajectory tracker, allowing for the MPC on a manifold to select inputs to track the leaders trajectory.

![control loop](figures/ControlLoop.png?raw=true)

The figure below depicts the basic architecture and information flow through the Seq2Seq architecture. A Transformer version of this is forthcoming.

![seq2seq](figures/seq2seq.png?raw=true)

## Environment Setup
We provide a Conda environment config for easy setup. Use
'''
conda env create -f config.yml
'''
to set up an environment with all of the required packages.

## Scripts

All scripts are stored in the scripts directory, under four distinct sub-categories. To utilize them, please copy them into the top level directory. We briefly list the scripts and a simplified description for each by subcategory below. We note that exact details on command-line arguments for the scripts can be found by running each script with the '-h' flag.

### scripts/general/
This directory contains scripts for generating the numerical simulation dataset, training and evaluating a neural network and plotting various model outputs. The scripts include:
- gen_dataset.py - Run numerical simulations and record the outputs into a dataset using the true trajectory values. All values are modified in the code in this script.
- run.py - Train a given neural network on the specified dataset. Several options for preprocessing, etc. are given.
- eval.py - Evaluate a model on a test dataset generated independently from the training set.
- plot.py - Generate graphics using a trained model and a given dataset.

### scripts/numerical_sim/
This directory contains scripts for interacting with a numerical simulation of the leader-follower convoy problem. These scripts are:
- control_loop.py - Run a single loop of the numerical simulation with a specified model and controller, calculating errors and plotting at the end.
- eval_control.py - Run and record a number of simulations for a specified model and controller for evaluation.
- results.py - Calculate errors and plot for prespecified results from the eval_control.py script.

### scripts/iekf/
This directory contains scripts for interacting with the dual IEKF trajectory tracking method.
- eval_iekf.py - Evaluate the IEKF method on the specified dataset.
- kalman.py - Run and plot the dual IEKF trajectory tracking method for a single run.
- eval_iekf.py - Run and collect evaluation data for the IEKF from simulation.

### scripts/bullet_sim/
This directory contrains scripts for interacting with the PyBullet simulation. These scripts are:
- spin_bullet_df.py - Generate a dataset for training the neural network from the pybullet simulations. Set up for the Scout Mini robot.
- bullet_sim.py - Run simulations in bullet, with an optional GUI. Defaults to only use GMPC. Plots trajetories at termination.
- eval_bullet.py - Run and record simulations in bullet for later evaluations.
- bullet_results - Calculate errors and plot for predefined bullet simulation files.

## Directory Structure

### Simulations
This directory contains the simulator files. convoy.py contains the interface for a simulation that spins up a random trajectory and allows for the control of a unicycle meant to follow the trajectory. It contains a subdriectory for bullet simulations.

### Controllers
This contains code for three controllers for the state-feedback loop(Geometric MPC, PID, standard MPC). It also contains a PID for generating data.

### Datasets
This contains the simulated data for training the internal model of the trajectory to follow. It also contains some utilities for formatting and filtering the generated data for use by the models.

### Trajectory Models
This contains the code for the various trajectory models. They are segmented into three subdirectories, wrappers for interfaces that work with the state-feedback loop, trainable for NN models and their losses, and IEKF for the IEKF implementations. 

### Common
This directory contains general utilities and common code for multiple structures. This includes transformations, code for the SE2 manifolds, plotting and generating directories, among other things.

### Models
This directory contains trained/saved neural network models. It may not exist prior to running the training at the moment, as I have not uploaded any models.

### bulletEval + controlEval
These directories contain the saved evaluation trajectories for simulated runs of model-controller pairs.

## Acknowledgements
We would like to thank the authors of the following repositories for their work.
- [Geometric Model Predictive Control](https://github.com/Garyandtang/GMPC-Tracking-Control) - We modify the controllers in this work to utilize the SE2 trajectories we estimate. Further, their simulation work was instrumental in setting up our own PyBullet simulations. 
- [Invariant Extended Kalman Filters](https://github.com/contagon/iekf/tree/master) - The work here was useful for setting up our IEKFs for baselines as well as for our NN methods warm start prediction.

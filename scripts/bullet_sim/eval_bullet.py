from simulations.convoy import convoy
import torch
import numpy as np
import matplotlib.pyplot as plt
from common.metrics import calc_rms_no_manifold, calc_rms, calc_box_minus_rms
import common.configs as config
import common.util as util
from trajectory_models.base import *
import argparse
from simulations.bullet_sim.main_single_tracking_task import *

parser = argparse.ArgumentParser(description='This script handles evaluating the various trajectory models in a state-feedback loop.')

parser.add_argument('--model', dest='model', metavar='model_name', default='neural-net',
                    help='Name of the model we wish to use. Valid options include ' + str(config.model_wrapper_list.keys()))

parser.add_argument('--relative_state', dest='rs', metavar='relative_state', default='True',
                    help='Whether to use a relative state for the model and control or not. Options are True or False')

parser.add_argument('--dev', dest='dev', metavar='dev', default='cpu',
                    help='Device we wish to use. Using argument auto automatically selects a GPU.')

parser.add_argument('--target_preproc', dest='preproc', metavar='preproc', default='CosSin',
                    help='Preprocessing used on the inital target variable for skip connections. Options include ' + str(config.preproc_list.keys()))

parser.add_argument('--load-path', dest='load_path', metavar='load-path', default='models/skipseq2seq/skipGRUDecoder/transfered/',
                    help='Path to the trained model we wish to use.')

parser.add_argument('--save-type', dest='save', metavar='save', default='val_',
                    help='Prefix for the saved model from val_ (for best validation loss), best_ (for best training loss) and \'\' for the final weights.')

parser.add_argument('--num-trials', dest='trials', metavar='num-trials', default='10',
                    help='Integer number of trials to run.')

parser.add_argument('--bot', dest='bot', metavar='robot', default='scout',
                    help='Robot URDF we wish to use. From list :' + str(config.bullet_models.keys()))

parser.add_argument('--dt', dest='dt', metavar='dt', default='0.02',
                    help='Float value of dt, or 1/control frequency')

parser.add_argument('--steps', dest='steps', metavar='steps', default='8000',
                    help='Integer value for number of timesteps.')

args = parser.parse_args()

# Validate selections for script and load everything
# Set up training device
if args.dev == 'auto':
    dev = util.get_device()
else:
    dev = args.dev

# Check the trained model
if args.model not in config.model_wrapper_list.keys():
    print('Invalid model: ' + args.model +'. Select from: ' + str(config.model_wrapper_list.keys()))
    exit(-1)

# Check and extract info for target/skip connection preprocessing
if args.preproc not in config.preproc_out_size.keys():
    print('Invalide preprocessing function selected. Choose from ' + str(config.preproc_list.keys()))
    exit(-1)

preproc_func = config.preproc_list[args.preproc]

# Get the relative state value
relative_state = True if args.rs=='True' else False

# Load the model using an external function
trajModel = config.load_traj_model(args.model, args.load_path, args.save, preproc_func, relative_state)

# Check the URDF
if args.bot not in config.bullet_models.keys():
    print('Invalid robot: ' + args.bot + '. Select from: ' + str(config.bullet_models.keys()))
    exit(-1)

robot = config.bullet_models[args.bot]

dt = float(args.dt)
num_steps = int(args.steps)

# Set up matrices for sampling noise
Q = np.array([[0.02, 0.0, 0.0],
              [0.0,  0.02,0.0],
              [0.0,  0.0,0.02]])
R = np.array([[0.02, 0.0],
              [0.0,  0.02]])

# Generate a path so save the runs
path = util.generate_timestamped_path('bulletEval/')


# Run N trials, with N being taken from the command line
numTrials = int(args.trials)

seed = 10
all_states = []
all_inputs = []
all_traj   = []
for i in range(numTrials):
    states, inputs, traj = bullet_tracking(trajModel, Q, R, robot, dt=dt, maxSteps=num_steps,seed=seed)
    # Save the output data
    os.makedirs(path + 'run_' + str(i) + '/')
    np.save(path + 'run_' + str(i) + '/states', states)
    np.save(path + 'run_' + str(i) + '/inputs', inputs)
    np.save(path + 'run_' + str(i) + '/traj', traj)
    all_states.append(states)
    all_traj.append(traj)
    all_inputs.append(inputs)
    seed += 1

# Print errors over runs so we can just use these directly
rms_error, l = calc_rms(all_states, all_traj)
print(rms_error)
rms_error, l = calc_box_minus_rms(all_states, all_traj)
print(rms_error)

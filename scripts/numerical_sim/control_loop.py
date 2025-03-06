from simulations.convoy import convoy
import torch
import numpy as np
import matplotlib.pyplot as plt
from common.metrics import calc_rms_no_manifold, calc_rms, calc_box_minus_rms
import common.configs as config
import common.util as util
from trajectory_models.base import *
import argparse
import random

parser = argparse.ArgumentParser(description='This script handles evaluating the various trajectory models in a state-feedback loop.')

parser.add_argument('--model', dest='model', metavar='model_name', default='neural-net',
                    help='Name of the model we wish to use. Valid options include ' + str(config.model_wrapper_list.keys()))

parser.add_argument('--control', dest='control', metavar='controller', default='pid',
                    help='Name of the controller we wish to use. Valid options include ' + str(config.controller_list.keys()))

parser.add_argument('--dev', dest='dev', metavar='dev', default='cpu',
                    help='Device we wish to use. Using argument auto automatically selects a GPU.')

parser.add_argument('--target_preproc', dest='preproc', metavar='preproc', default='CosSin',
                    help='Preprocessing used on the inital target variable for skip connections. Options include ' + str(config.preproc_list.keys()))

parser.add_argument('--relative_state', dest='rs', metavar='relative_state', default='True',
                    help='Whether to use a relative state for the model and control or not. Options are True or False')

parser.add_argument('--load-path', dest='load_path', metavar='load-path', default='models/skipseq2seq/skipGRUDecoder/gmpc/',
                    help='Path to the trained model we wish to use.')

parser.add_argument('--debug', dest='debug', metavar='debug', default='False',
                    help='Flag to run in debug mode. True runs in debug, anything else does not.')

parser.add_argument('--save-type', dest='save', metavar='save', default='val_',
                    help='Prefix for the saved model from val_ (for best validation loss), best_ (for best training loss) and \'\' for the final weights.')

args = parser.parse_args()

np.random.seed(0)
random.seed(0)

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

# Check for a valid controller
if args.control not in config.controller_list.keys():
    print('Invalid controller choice. Choose from ' + str(config.controller_list.keys()))
    exit(-1)

# Load the controller with appropriate relative state setting
control = config.controller_list[args.control](relativeState=relative_state)

debug = args.debug == 'True'

# Set up matrices for sampling noise
Q = np.array([[0.02, 0.0, 0.0],
              [0.0,  0.02,0.0],
              [0.0,  0.0,0.002]])
R = np.array([[0.02, 0.0],
              [0.0,  0.02]])



# Initialize the simulation
simulator = convoy(trajModel, control, Q, R)
simulator.reset()
simulator.init()

# Run the simulator for a trajectory
states, inputs, traj = simulator.simulate(debug=debug)

# Plotting stuff
plt.clf()

plt.plot(states[:,0], states[:,1],'b-', label="States")
plt.plot(states[0,0], states[0,1],'b*')
plt.plot(traj[:,0], traj[:,1], 'r-', label='True Traj')
plt.plot(traj[0,0], traj[0,1], 'r*')
plt.legend()
plt.show()

plt.clf()

# Calculate RMS error
rms_error, l = calc_rms([states], [traj])
print(rms_error)
rms_error, l = calc_box_minus_rms([states], [traj])
print(rms_error)

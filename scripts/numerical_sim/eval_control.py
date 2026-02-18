from simulations.convoy import convoy
import torch
import numpy as np
import matplotlib.pyplot as plt
from common.metrics import calc_rms,calc_box_minus_rms
import common.configs as config
import common.util as util
from trajectory_models.base import *
import argparse
import os
import signal
from common.util import RRTTimeoutHandler


parser = argparse.ArgumentParser(description='This script handles evaluating the various trajectory models in a state-feedback loop.')

parser.add_argument('--model', dest='model', metavar='model_name', default='neural-net',
                    help='Name of the model we wish to use. Valid options include ' + str(config.model_wrapper_list.keys()))

parser.add_argument('--control', dest='control', metavar='controller', default='gmpc',
                    help='Name of the controller we wish to use. Valid options include ' + str(config.controller_list.keys()))

parser.add_argument('--dev', dest='dev', metavar='dev', default='cpu',
                    help='Device we wish to use. Using argument auto automatically selects a GPU.')

parser.add_argument('--target_preproc', dest='preproc', metavar='preproc', default='noPreProc',
                    help='Preprocessing used on the inital target variable for skip connections. Options include ' + str(config.preproc_list.keys()))

parser.add_argument('--relative_state', dest='rs', metavar='relative_state', default='True',
                    help='Whether to use a relative state for the model and control or not. Options are True or False')

parser.add_argument('--load-path', dest='load_path', metavar='load-path', default='models/skipseq2seq/skipGRUDecoder/gmpc/',
                    help='Path to the trained model we wish to use.')

#parser.add_argument('--load-path', dest='load_path', metavar='load-path', default='models/skipseq2seq/skipGRUDecoder/mse/',
#                    help='Path to the trained model we wish to use.')


parser.add_argument('--debug', dest='debug', metavar='debug', default='False',
                    help='Flag to run in debug mode. True runs in debug, anything else does not.')

parser.add_argument('--save-type', dest='save', metavar='save', default='val_',
                    help='Prefix for the saved model from val_ (for best validation loss), best_ (for best training loss) and \'\' for the final weights.')

parser.add_argument('--num-trials', dest='trials', metavar='num-trials', default='70',
                    help='Integer number of trials to run.')

parser.add_argument('--disturbance-type', dest='disturbance_type', metavar='disturbance_type', default=None,
                    help='Type of disturbance to apply: "constant" or "steering_bias" or None.')
parser.add_argument('--disturbance-value', dest='disturbance_value', metavar='disturbance_value', default="0.1,0.0,0.0",
                    help='Value for the disturbance. For constant: comma-separated (e.g. "0.1,0.0"). For steering_bias: float.')

parser.add_argument('--noise-type', dest='noise_type', metavar='noise_type', default='normal',
                    help='Type of observation noise: "normal", "uniform", or "anisotropic".')

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
              [0.0,  0.0,0.02]])
R = np.array([[0.02, 0.0],
              [0.0,  0.02]])

# Parse disturbance_value if provided
if args.disturbance_value is not None:
    if args.disturbance_type == 'constant':
        disturbance_value = np.array([float(x) for x in args.disturbance_value.split(',')])
    elif args.disturbance_type == 'steering_bias':
        disturbance_value = float(args.disturbance_value)
    else:
        disturbance_value = None
else:
    disturbance_value = None

# Generate a path so save the runs
path = util.generate_timestamped_path('controlEval/')

# Initialize the simulation
simulator = convoy(trajModel, control, Q, R, 
                  disturbance_type=args.disturbance_type, 
                  disturbance_value=disturbance_value,
                  noise_type=args.noise_type)

numTrials = int(args.trials)

seed = 0 
#seed = 41
all_states = []
all_inputs = []
all_traj   = []

def eval_seed(simulator, seed):
    print('simulating')
    simulator.reset(seed=seed)
    print('Reset complete')
    simulator.init()
    print('Running Sim')
    states, inputs, traj = simulator.simulate()
    os.makedirs(path + 'run_' + str(seed) + '/')
    np.save(path + 'run_' + str(seed) + '/states', states)
    np.save(path + 'run_' + str(seed) + '/inputs', inputs)
    np.save(path + 'run_' + str(seed) + '/traj', traj)
    return states, traj, inputs

i = 0
while i < numTrials:
    signal.signal(signal.SIGALRM, RRTTimeoutHandler) 
    signal.alarm(800)
    try:
        states, traj, inputs = eval_seed(simulator, seed)
        #all_states.append(states)
        #all_traj.append(traj)
        #all_inputs.append(inputs)
    except Exception as exc:
        i -= 1
        print(exc)

    seed += 1
    print(i,seed)
    i+=1

    signal.alarm(0)
    
print(i, seed)



#rms_error, l = calc_rms(all_states, all_traj)
#print(rms_error)
#rms_error, l = calc_box_minus_rms(all_states, all_traj)
#print(rms_error)




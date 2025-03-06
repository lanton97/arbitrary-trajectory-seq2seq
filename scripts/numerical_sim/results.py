from common.metrics import *
import common.configs as config
import common.util as util
from datasets.dataset import convoyDataset
from trajectory_models.base import *
from simulations.convoy import convoy
import torch
import numpy as np
from common.metrics import calc_rms_no_manifold, calc_rms, calc_box_minus_rms
import os
from common.plotting import *

path_prefix = 'controlEval/'
path_suffix = 'run_' 
num_trials = 10
paths = ['gmpc/',  'pid/',  'true_gmpc/', 'true_pid/']

for path in paths:

    all_states = []
    all_traj = []
    all_inputs = []
    for i in range(num_trials):
        states = np.load(path_prefix + path + path_suffix + str(i) + '/states.npy')
        traj = np.load(path_prefix + path + path_suffix + str(i) + '/traj.npy')
        inputs = np.load(path_prefix + path + path_suffix + str(i) + '/inputs.npy')
        all_states.append(states)
        all_traj.append(traj)
        all_inputs.append(inputs)


    plot_traj_error(states, traj)
    plot_control_effort(states, traj, inputs)
    print(path + ' RMS Error')
    rms_error, l = calc_rms(all_states, all_traj)
    print(rms_error)
    print(path + ' Boxminus RMS Error')
    rms_error, l = calc_box_minus_rms(all_states, all_traj)
    print(rms_error)
    print(path + ' RMS Error w/o theta')
    rms_error, l = calc_rms_no_manifold(all_states, all_traj)
    print(rms_error)


states = np.load(path_prefix + paths[0] + path_suffix + str(0) + '/states.npy')
traj = np.load(path_prefix + paths[0] + path_suffix + str(0) + '/traj.npy')
inputs = np.load(path_prefix + paths[0] + path_suffix + str(0) + '/inputs.npy')

plot_control_effort(states, traj, inputs)


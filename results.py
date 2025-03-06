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
from common.debug import *

path_prefix = 'bulletEval/'
path_suffix = 'run_' 
num_trials = 10
paths = ['seq2seq_scout/', 'seq2seq_turtle/', 'iekf_scout/', 'iekf_turtle/', 'true_scout/', 'true_turtle/']

for path in paths:

    all_states = []
    all_traj = []
    all_inputs = []
    print(path)
    for i in range(num_trials):
        states = np.load(path_prefix + path + path_suffix + str(i) + '/states.npy')
        traj = np.load(path_prefix + path + path_suffix + str(i) + '/traj.npy')
        inputs = np.load(path_prefix + path + path_suffix + str(i) + '/inputs.npy')
        all_states.append(states)
        all_traj.append(traj)
        all_inputs.append(inputs)
        if i == 4:
            plot_traj_error(states, traj)
            plot_control_effort(states, traj, inputs)
     
            errorInd = detectErrorRegion(states, traj,thresh=0.1)
     
            plt.plot(states[:,0], states[:,1],'b-', label="States")
            plt.plot(states[0,0], states[0,1],'b*')
            plt.plot(traj[:,0], traj[:,1], 'r-', label='True Traj')
            plt.plot(traj[0,0], traj[0,1], 'r*')
            plt.plot(states[errorInd,0], states[errorInd,1], 'g*', label='State marked as error beginning.')
            plt.legend()
            plt.show()




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


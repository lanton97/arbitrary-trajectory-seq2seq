from simulations.iekf_convoy import *
from simulations.dual_iekf import *
from common.se2 import *
from copy import deepcopy
from common.transformations import *
from common.metrics import *
from controllers.gmpc import *
from common import util
import os

numTrials = 10

path = util.generate_timestamped_path('controlEval/')
# Run the Dual IEKF convoy without plotting
sim = dual_iekf_convoy(control=GeometricMPC)

all_states = []
all_traj   = []
all_pred_traj = []
all_true_traj = []
for i in range(numTrials):
    sim.reset()
    ego, leader = sim.simulate(plot=False)
    true_state, _, des_traj = ego

    os.makedirs(path + 'run_' + str(i) + '/')
    np.save(path + 'run_' + str(i) + '/states', true_state)
    np.save(path + 'run_' + str(i) + '/traj', des_traj)
    all_states.append(true_state[0])
    all_traj.append(des_traj[0])

    pred_state, true_state = leader
    np.save(path + 'run_' + str(i) + '/pred_rel', pred_state)
    np.save(path + 'run_' + str(i) + '/true_rel', true_state)
    all_pred_traj.extend(pred_state)
    all_true_traj.extend(true_state)


# CTRL RMSE
error,_ = calc_rms(all_states, all_traj)
print('Control RMSE: ' + str(error))

# CTRL Boxminus RMSE
error,_ = calc_box_minus_rms(all_states, all_traj)
print('Control Boxminus RMSE: ' + str(error))

# Leader Pred RMSE
error,_ = calc_rms(all_true_traj, all_pred_traj)
print('Leader Pred RMSE: ' + str(error))

# Leader Pred Boxminus RMSE
error,_ = calc_box_minus_rms(all_true_traj, all_pred_traj)
print('Leader Pred Boxminus RMSE: ' + str(error))


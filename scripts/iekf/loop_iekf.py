from simulations.iekf_convoy import *
from simulations.dual_iekf import *
from common.se2 import *
from copy import deepcopy
from common.transformations import *
from common.metrics import *
from controllers.gmpc import *
from common import util
import os

numTrials = 100
seed = 0

path = util.generate_timestamped_path('controlEval/')
# Run the Dual IEKF convoy without plotting
sim = dual_iekf_convoy(control=GeometricMPC)

def eval_seed(simulator):
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

i = 0
while i < numTrials:
    i += 1
    signal.signal(signal.SIGALRM, RRTTimeoutHandler) 
    signal.alarm(800)

    try:
        simulator.reset(seed, seed)

    except Exception as exc:
        i -= 1
        continue
        print(exc)


    eval_seed(simulator, seed)

from trajectory_models.iekf.moving_observer_unicycle import *
from trajectory_models.iekf.dual_iekf import *
import matplotlib.pyplot as plt 
from common.se2 import *
from datasets.dataset_utils import *
from datasets.dataset import convoyDataset
from common.metrics import *
import numpy as np

def getInputs(obs, iekfFeatures):
    # Split IEKF features and observations into their constituent parts
    ego_us = iekfFeatures[:,-4:-2]
    leader_us = iekfFeatures[:,-2:]
    ego_z = iekfFeatures[:,:3]
    ego_x0 = iekfFeatures[0,:3]
    leader_x0 = obs[0,0:3]
    leader_z = iekfFeatures[:,-7:-4]

    # Convert to manifold trajectories
    zeMan = convertTrajToManifolds(ego_z)
    zlMan = convertTrajToManifolds(leader_z)

    ego_z = []
    leader_z = []

    b = np.array([0,0,1])
    for i in range(len(zeMan)):
        # Get the measurements from the trajectory matrices
        ego_z.append(np.array(zeMan[i].t_matrix)@b)
        leader_z.append(np.array(zlMan[i].t_matrix)@b)

    ego_z = np.array(ego_z)
    leader_z = np.array(leader_z)

    return ego_us, leader_us, ego_x0, leader_x0, ego_z, leader_z


def getPred(ds, idx):
    obs, traj = ds[idx]
    iekfFeatures = ds.getIEKFItems(idx)

    # setup system
    Q = np.diag([.001, 0, .1])
    R = np.diag([.001, .001])
    dt = 0.1
    sys = MovingObserverUnicycle(Q, R, dt)

    ego_us, leader_us, ego_x0, leader_x0, ego_z, leader_z = getInputs(obs, iekfFeatures)

    # Run the iekf
    iekf = DualIEKFTracker(sys, ego_x0, np.eye(3), leader_x0, np.eye(3))
    emus, esigmas, lmus, lsigmas = iekf.iterate(ego_us, ego_z, leader_us, leader_z)

    relative_traj, base_traj = iekf.translateToCoordinateBase(-1, emus, lmus)

    return torch.from_numpy(relative_traj), torch.from_numpy(lsigmas)


DS = convoyDataset(file_path='datasets/test.csv',leader_speed=True, iekf=True)


true_trajs = []
pred_trajs = []
for i in range(len(DS)): 
    _, target = DS[i]
    q_hat, pred = getPred(DS, i)
    pred_trajs.append(q_hat)
    true_trajs.append(target)
rms, errors = calc_box_minus_rms(pred_trajs, true_trajs)
print(rms)
print(errors)

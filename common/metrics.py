import numpy as np
from common.se2 import *

# Calculate the Rn space RMS error
def calc_rms(trajs, true_trajs):
    rms_errors = []
    variances = []
    for i in range(len(trajs)):
        traj = convertTrajToManifolds(trajs[i])
        true_traj = np.array(true_trajs[i])
        rms = 0
        for j in range(len(traj)):
            pos = traj[j].vector.listView
            rms += (np.linalg.norm(true_traj[j] - pos))
        rms = np.sqrt(rms / len(traj))
        rms_errors.append(rms)

    rms = np.mean(rms_errors)
    return rms, rms_errors

# Calculate the RMS error without taking into account the angle
def calc_rms_no_manifold(trajs, true_trajs):
    rms_errors = []
    variances = []
    for i in range(len(trajs)):
        traj = np.array(trajs[i])
        true_traj = np.array(true_trajs[i])
        rms = 0
        for j in range(len(traj)):
            pos = traj[j]
            rms += (np.linalg.norm(true_traj[j,:2] - pos[:2]))
        rms = np.sqrt(rms / len(traj))
        rms_errors.append(rms)

    rms = np.mean(rms_errors)
    return rms, rms_errors

# Calculate the RMS error using boxminus instead of a vector space subtraction
def calc_box_minus_rms(trajs, true_trajs):
    rms_errors = []
    variances = []
    for i in range(len(trajs)):
        traj = convertTrajToManifolds(trajs[i])
        true_traj = convertTrajToManifolds(true_trajs[i])
        rms = 0
        for j in range(len(traj)):
            rms += (np.linalg.norm(BoxMinus(true_traj[j], traj[j])))
        rms = np.sqrt(rms / len(traj))
        rms_errors.append(rms)

    rms = np.mean(rms_errors)
    return rms, rms_errors

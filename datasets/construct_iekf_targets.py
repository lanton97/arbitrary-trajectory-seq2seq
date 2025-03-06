from trajectory_models.iekf.iekf import * 
from trajectory_models.iekf.unicycle_wrapper import * 
import numpy as np
from common.se2 import convertTrajToManifolds

def createMeasurementsFromPose(poses):
    b = np.array([0,0,1])
    mans = convertTrajToManifolds(poses)
    views = []
    for man in mans:
        views.append(man.t_matrix.numpy() @ b)
    return views


def constructIEKFTargets(iekf_items):
    trgs = []
    for item in iekf_items:
        poses, inps, obs = item
        trg = constructIEKFTarget(poses, inps, obs)
        trgs.append(trg)
    return trgs

def constructIEKFTarget(poses, inps,obs):
    x0 = poses[0]
    Q = np.array([[0.05, 0., 0.],
                  [0., 0.05, 0.],
                  [0., 0., 0.05]])
    R = np.array([[0.05, 0.],
                  [0., 0.05]])
    system = UnicycleWrapper(Q, R, 0.1)
    iekf = InvariantEKF(system, x0, Q)

    z = createMeasurementsFromPose(poses)
    mu, sig = iekf.iterate(inps, z)
    refPos = mu[-1]

    initPos = iekf.translateSinglePointToCoordinateBase(mu[0], refPos, iekf.sys.vecToSE2(obs[0]))
    initPos = torch.unsqueeze(manifoldToVector(initPos), 0)
    

    return initPos





from .model_wrapper import ModelWrapper
from trajectory_models.iekf.unicycle_wrapper import *
from trajectory_models.iekf.moving_observer_unicycle import *
from trajectory_models.iekf.iekf import InvariantEKF
from common.se2 import *
from trajectory_models.iekf.dual_iekf import DualIEKFTracker

class DualIEKFWrapper(ModelWrapper):
    def __init__(self,
                 dt=0.02
                 ):
        self.obs = []
        self.us = []
        self.x0 = None
        self._traj = []
        self._covs = []
        Q = np.diag([.001, 0, .1])
        R = np.diag([.001, .001])
        self.b = np.array([0,0,1])
        self.sys = MovingObserverUnicycle(Q, R, dt)

    @property
    def traj(self):
        return self._traj, self._covs

    @property
    def window(self):
        us = torch.Tensor(self.us[-100:])
        obs = torch.Tensor(self.obs[-100:])
        ths = torch.unsqueeze(obs[:,2], 1)
        window = torch.cat([obs[:,:2], torch.cos(ths), torch.sin(ths), us], axis=1)
        
        return torch.unsqueeze(window, 0)

    def push_init(self, init_pos, egoTraj, egoUs, leaderUs, ):
        self.obs.extend(egoTraj)
        self.us = egoUs
        self.leaderUs = leaderUs 
        self.x0 = init_pos
        egoPos = [init_pos for i in range(len(egoTraj))]
        zeMan = convertTrajToManifolds(egoPos)
        zlMan = convertTrajToManifolds(self.obs)


        ze = []
        zl = []

        for i in range(len(zeMan)):
            ze.append(np.array(zeMan[i].t_matrix)@self.b)
            zl.append(np.array(zlMan[i].t_matrix)@self.b)

        ze = np.array(ze)
        zl = np.array(zl)

        self.iekf = DualIEKFTracker(self.sys, self.x0, np.eye(3), egoTraj[0], np.eye(3))
        egoMus, egoCov, leaderMus, leaderCov = self.iekf.iterate(self.us, ze, self.leaderUs, zl)
        ref_point = self.iekf.egoTracker.mu
        est_leader_traj    =  [manifoldToVector(self.iekf.translateSinglePointToCoordinateBase(eMu, ref_point, lMu)).numpy() for eMu, lMu in zip(self.iekf.egoTracker.mus[-100:],self.iekf.leaderTracker.mus[-100:])]

        self._traj = est_leader_traj

    def step(self, egoView, newReading, egoU, leaderUs, trueTraj):

        self.obs.append(egoView)

        # Calculate observations from the position and 
        manifoldEgoPos = self.iekf.system.vecToSE2(newReading)
        manifoldEgoObs = manifoldEgoPos @ self.b
        manifoldLeadPos = self.iekf.system.vecToSE2(egoView)
        manifoldLeadObs = manifoldLeadPos @ self.b

        egoMus, egoCov, leaderMus, leaderCov = self.iekf.step(egoU, manifoldEgoObs, leaderUs, manifoldLeadObs)

        ref_point = self.iekf.egoTracker.mu
        est_leader_traj    =  [manifoldToVector(self.iekf.translateSinglePointToCoordinateBase(eMu, ref_point, lMu)).numpy() for eMu, lMu in zip(self.iekf.egoTracker.mus[-100:],self.iekf.leaderTracker.mus[-100:])]

        self._traj = np.array(est_leader_traj)

        self._covs = egoCov

        return self._traj, self._covs

    def reset(self):
        self.obs = []
        self.us = []
        self.x0 = None
        self._traj = []
        self._covs = []


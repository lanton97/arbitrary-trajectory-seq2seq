from .model_wrapper import ModelWrapper
from trajectory_models.iekf.unicycle_wrapper import *
from trajectory_models.iekf.iekf import InvariantEKF
from trajectory_models.iekf.iekf import InvariantEKF
from common.se2 import *
import torch
import common.util as util
from common.preproc import noPreProc

# Wrap the neural network model for an interface with the control loop
class NeuralNetworkWrapper(ModelWrapper):
    def __init__(self,
                 nn_model,
                 target_preproc=noPreProc,
                 leader_inp=True,
                 dt=0.02,
                 train_dt=0.1
                 ):
        self.obs = []
        self.us = []
        self.leader_us = []
        self.x0 = None
        self._traj = []
        self._covs = []
        self.trg_preproc = target_preproc
        # Used for the IEKF tracking of the ego vehicle 'global' position
        Q = np.diag([.001, 0, .1])
        R = np.diag([.001, .001])
        self.dt = dt
        # Ratio of training sampling to run time sampling
        # Used to modulate the inputs
        self.dt_ratio = dt/train_dt
        self.offset = 100
        self.b = np.array([0,0,1])
        self.sys = UnicycleWrapper(Q, R, dt)
        # Save a loaded NN model
        self.model = nn_model
        self.leader_inp = leader_inp

    # Return the trajectory and covariances
    @property
    def traj(self):
        if torch.is_tensor(self._traj):
            return self._traj.detach().numpy(), self._covs.detach().numpy()

        return self._traj, self._covs

    # Get the last 100 input items for NN inputs
    @property
    def window(self):
        us = torch.Tensor(self.us[-100:])*self.dt_ratio
        obs = torch.Tensor(self.obs[-100:])
        leader_us = torch.Tensor(self.leader_us[-100:])*self.dt_ratio
        ths = torch.unsqueeze(obs[:,2], 1)
        if self.leader_inp:
            window = torch.cat([obs[:,:2], torch.cos(ths), torch.sin(ths), us, leader_us], axis=1)
        else:
            window = torch.cat([obs[:,:2], torch.cos(ths), torch.sin(ths), us], axis=1)
        
        return torch.unsqueeze(window, 0)

    # Push the initial 100 timesteps
    def push_init(self, init_pos, egoTraj, egoUs, leaderUs):
        self.obs.extend(egoTraj)
        self.us.extend(egoUs)
        #leaderUs = [u*self.dt_ratio for u in leaderUs]
        self.leader_us.extend(leaderUs)
        self.x0 = init_pos
        self.iekf = InvariantEKF(self.sys, self.x0, np.eye(3))
        #initPos = torch.Tensor.unsqueeze(manifoldToVector(init_pos), 0)
        initPos = np.expand_dims(init_pos, 0)
        initPos = np.expand_dims(initPos, 0)
        initPos = torch.Tensor(initPos)
        trg_input = self.trg_preproc(initPos)

        # Get our NN outputs
        self._traj, self._covs = self.model(self.window, trg_input)
        self._traj = torch.squeeze(self._traj)


    def step(self, egoView, newReading, egoU, leaderInp, trueTraj):
        # Store the ego view
        self.obs.append(egoView)
        self.us.append(egoU)
        self.leader_us.append(leaderInp)

        # Step the IEKF for the ego global position
        manifoldPos = SE2(vec=newReading)
        manifoldObs = (manifoldPos.t_matrix @ self.b).numpy()
        self.iekf.step(egoU, manifoldObs)

        refPos = self.iekf.mus[-1]

        #initPos = self.iekf.translateSinglePointToCoordinateBase(egoPos, self.sys.vecToSE2(self.obs[-100]))
        initPos = self.iekf.translateSinglePointToCoordinateBase(self.iekf.mus[-99:][0], refPos, self.sys.vecToSE2(self.obs[-99]))
        # TODO: Temporary while implementing a manifold controller
        initPos = torch.Tensor.unsqueeze(manifoldToVector(initPos), 0)
        initPos2 = torch.Tensor.unsqueeze(torch.Tensor(global2LocalCoords(newReading, trueTraj[0])), dim=0)
        initPos = torch.Tensor.unsqueeze(initPos, 0)

        trg_input = self.trg_preproc(initPos)


        # Get our NN outputs
        self._traj, self._covs = self.model(self.window, trg_input)
        self._traj = torch.squeeze(self._traj)

        return self.traj


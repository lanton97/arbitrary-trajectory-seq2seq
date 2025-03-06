from abc import ABC, abstractmethod

# Simple abstract class for trajectory model interfaces with
# the control loop
class ModelWrapper(ABC):
    def __init__(self):
        self.obs = []
        self.x0 = None
        self.us = []
        self._traj = []
        self._covs = []

    @property
    def traj(self):
        return np.array(self._traj)

    def reset(self):
        self.us = []
        self.obs = []
        self.x0 = None
        self._traj = []
        self.leaderUs = []
        self._covs = []

    def push_init(self, init_pos, egoTraj, egoUs, leaderUs):
        self.us.extend(egoUs)
        self.obs.extend(egoTraj)
        self.x0 = init_pos

    @abstractmethod
    def step(self, egoView, newReading, u, leaderInp, trajectoryInfo):
        pass




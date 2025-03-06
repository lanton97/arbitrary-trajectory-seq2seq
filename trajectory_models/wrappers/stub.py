from .model_wrapper import ModelWrapper
import numpy as np
from common.transformations import global2LocalCoords

class StubWrapper(ModelWrapper):
    def __init__(self, relative_state=True):
        self.obs = []
        self.x0 = None
        self._traj = []
        self.relative_state=relative_state

    @property
    def traj(self):
        return np.array(self._traj)

    def push_init(self, init_pos, egoTraj, egoUs, leaderUs):
        self.obs.extend(egoTraj)
        self.x0 = init_pos

    def step(self, egoView, newReading, inp, leaderInp, trueTraj):
        self._traj = trueTraj
        if self.relative_state:
            traj = [global2LocalCoords(newReading, pos, shiftToPiRange=False) for pos in trueTraj]
            self._traj = traj

        return np.array(self._traj), None


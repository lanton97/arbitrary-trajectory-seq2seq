import numpy as np
import time
from roboticstoolbox.mobile.drivers import VehicleDriverBase
from spatialmath import SE2, base
from abc import ABC, abstractmethod

# A driver class to control the unicycle model using IO linearization
# For control by exact feedback linearization
class ControllerWrapper(VehicleDriverBase):
    def __init__(
        self,
        workspace,
        trajectory,
        speed=2,
        seed=0,
        dt=0.1,
        vehicle=None
    ):

        if hasattr(workspace, "workspace"):
            # workspace can be defined by an object with a workspace attribute
            self._workspace = base.expand_dims(workspace.workspace)
        else:
            self._workspace = base.expand_dims(workspace)

        self._speed = speed
        self._goal_marker = None
        self._traj = trajectory
        self._t = 0
        self._dt=dt
        self._last_v = np.array([0.0,0.0])

        self._seed = seed
        self.verbose = True
        self._veh = vehicle
        self.inputs = []
        self.states = []

    def init(self, ax=None):
        pass

    def reset(self):
        pass

    def __str__(self):
        """%RandomPath.char Convert to string
        %
        % s = R.char() is a string showing driver parameters and state in in
        % a compact human readable format."""
        pass

    @property
    def workspace(self):
        """
        Size of robot driving workspace

        :return: workspace bounds [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Returns the bounds of the workspace as specified by constructor
        option ``workspace``
        """
        return self._workspace

    def set_traj(self, trajectory):
        self._traj = trajectory

    @abstractmethod
    def demand(self, s, traj, noise):
        pass
    
    # Reset the vehicles noise matrices for a new run
    def reset(self,):
        self.states = []
        self.inputs = []

    # Return our view of states with noise and the inputs
    def get_run_stats(self):
        return self.states, self.inputs

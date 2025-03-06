import numpy as np
import time
from roboticstoolbox.mobile.drivers import VehicleDriverBase
from spatialmath import SE2, base

# Select completely random inputs
# This is being done to see if this improves the coverage of off policy testing
class RandomController(VehicleDriverBase):
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

        self._random = np.random.default_rng(seed)
        self._seed = seed
        self.verbose = True
        self._veh = vehicle

        self.inputs = []
        self.states = []

    def init(self, ax=None):
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

    def demand(self):
        """
        Compute speed and heading for timestep
            %
            % [SPEED,STEER] = R.demand() is the speed and steer angle to
            % drive the vehicle toward the next waypoint.  When the vehicle is
            % within R.dtresh a new waypoint is chosen.
            %
            % See also Vehicle."""

        speed = self._speed


        q = self._veh._x + np.random.normal(loc=0.0, scale = 0.02, size=(3))
        self.states.append(q)
        # sample a random value uniformly
        v = np.random.uniform(low=-3.0, high=3.0, size=(2))
        self.inputs.append(v)

        return v

    def reset(self):
        self.states = []
        self.inputs = []

    def get_run_stats(self):
        return self.states, self.inputs

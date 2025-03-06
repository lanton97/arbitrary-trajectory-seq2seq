import numpy as np
import time
from roboticstoolbox.mobile.drivers import VehicleDriverBase
from spatialmath import SE2, base

# A driver class to control the unicycle model using IO linearization
# For control by exact feedback linearization
class UnicycleController(VehicleDriverBase):
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
        self.K = np.array([[1.5, 0.0],
                           [0.0, 1.5]])
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


        # Get the state and add noise
        q = self._veh._x + np.random.normal(loc=0.0, scale = 0.02, size=(3))
        self.states.append(q)
        theta = q[2]
        # Calculate the x and y direction velocity components
        zvel = np.array( [np.cos(theta)*speed, np.sin(theta)*speed] )

        # First, compute our error
        qd_pos = self._traj.eval((self._t)*self._dt)
        e = self.compute_error(q[:], qd_pos)


        # Compute our "new input", we use eta here
        eta = self.K.dot(e)
        # Compute the mapping of our eta to our control signals
        eta_to_v_trnsfrm = self.compute_eta_v_mapping(q)

        # Compute the control signals
        # Also add noise
        v = self._last_v + eta_to_v_trnsfrm.dot(eta) +np.random.normal(loc=0.0, scale = 0.02, size=(2))
        self._t +=1
        self.inputs.append(v)

        return v

    # Compute the nonlinear IO mapping
    def compute_eta_v_mapping(self,q):
        term_0_0 =  np.cos(q[2])
        term_0_1 =  np.sin(q[2])
        term_1_0 =  -np.sin(q[2])/self._veh._W
        term_1_1 =  np.cos(q[2])/self._veh._W
        eta_to_v_trnsfrm = np.array([[term_0_0, term_0_1], [term_1_0, term_1_1]])
        return eta_to_v_trnsfrm

    # Calculate our tracking error
    def compute_error(self, q, qd):
        x_error = qd[0] - (q[0])
        y_error = qd[1] - (q[1])
        e = np.array([x_error, y_error])
        return e
    
    # Reset the vehicles noise matrices for a new run
    def reset(self,):
        self.states = []
        self.inputs = []

    # Return our view of states with noise and the inputs
    def get_run_stats(self):
        return self.states, self.inputs

import numpy as np
import time
from roboticstoolbox.mobile.drivers import VehicleDriverBase
from spatialmath import SE2, base
from .wrapper import *

# A driver class to control the unicycle model using IO linearization
# For control by exact feedback linearization
class PIDStub(ControllerWrapper):
    def __init__(
        self,
        speed=2,
        seed=0,
        dt=0.1,
        vehicle=None,
        relativeState=False
    ):

        self._speed = speed
        self._goal_marker = None
        self._t = 0
        self._dt=dt
        self._last_v = np.array([0.0,0.0])

        self._seed = seed
        self.verbose = True
        self._veh = vehicle
        self.inputs = []
        self.states = []
        self.K = np.array([[1.5, 0.0],
                           [0.0, 1.5]])
        self._W = 1.0
        self.relativeState = relativeState

    def init(self, initObs):
        pass

    def demand(self, s, traj, noise):
        """
        Compute speed and heading for timestep
            %
            % [SPEED,STEER] = R.demand() is the speed and steer angle to
            % drive the vehicle toward the next waypoint.  When the vehicle is
            % within R.dtresh a new waypoint is chosen.
            %
            % See also Vehicle."""

        speed = self._speed
        traj = traj[0]
        if self.relativeState:
            s = np.array([0.0,0.0,0.0])


        # Get the state and add noise
        self.states.append(s)
        theta = s[2]
        # Calculate the x and y direction velocity components
        zvel = np.array( [np.cos(theta)*speed, np.sin(theta)*speed] )

        # First, compute our error
        e = self.compute_error(s[:], traj)


        # Compute our "new input", we use eta here
        eta = self.K.dot(e)
        # Compute the mapping of our eta to our control signals
        eta_to_v_trnsfrm = self.compute_eta_v_mapping(s)

        # Compute the control signals
        # Also add noise
        v = self._last_v + eta_to_v_trnsfrm.dot(eta)
        v += np.random.multivariate_normal(np.zeros(2), noise)
        self.inputs.append(v)

        return self.make_control_callable(v)

    def make_control_callable(self, u):
        func = lambda obj, t, s : u
        return func

    # Compute the nonlinear IO mapping
    def compute_eta_v_mapping(self,q):
        term_0_0 =  np.cos(q[2])
        term_0_1 =  np.sin(q[2])
        term_1_0 =  -np.sin(q[2])/self._W
        term_1_1 =  np.cos(q[2])/self._W
        eta_to_v_trnsfrm = np.array([[term_0_0, term_0_1], [term_1_0, term_1_1]])
        return eta_to_v_trnsfrm

    # Calculate our tracking error
    def compute_error(self, q, qd):
        x_error = qd[0] - (q[0])
        y_error = qd[1] - (q[1])
        e = np.array([x_error, y_error])
        return e

import numpy as np
import cvxpy as cp
import time
from roboticstoolbox.mobile.drivers import VehicleDriverBase
from roboticstoolbox.mobile import Unicycle
from spatialmath import SE2, base
from controllers.wrapper import *
import math
from scipy.optimize import minimize, LinearConstraint

# Step of the pendulum system
def unicycle_step(x, u):
    unicycle = Unicycle()
    xNew = unicycle.f(x, u)

    return xNew

def to_flat(nested_list):
    '''
    flatten a list of lists
    '''
    flat_list = []
    for l in nested_list:
        for e in l:
            flat_list.append(e)
    return flat_list


def two_dim_from_flat(flat_list, dim):
    '''
    here the second argument is a nested list in the desired shape -- values not relevant only shape
    '''
    nested_list = []
    idx = 0
    for i in range(0, len(flat_list), dim):
        nested_list.append(flat_list[i:i+dim])
    return nested_list


# Cost function to be minimized
def mpc_cost(u, u_ini, x0, traj,  Q, R, N):

    # Initialise cost = 0 and states to current measured states
    cost = 0
    x = x0 

    # Simulate the next N steps of the system
    for idx in range(0, N, 2):
        inp = [u[idx], u[idx+1]]
        x = unicycle_step(x, inp)
        inp = np.array(inp)

        cost +=  (x[:2] - traj[idx][:2]).T@Q @ (x[:2] - traj[idx][:2])
        cost += (inp).T@R @ inp
    return cost

# Adapted from https://github.com/tsender/me561-unicycle-mpc/blob/master/turtlebot_mpc/src/turtlebot_mpc/unicycle_mpc.py
class SimpleMPC(ControllerWrapper):
    def __init__(
        self,
        speed=2,
        seed=0,
        dt=0.1,
        horizon=25,
        vehicle=None,
        relativeState=False
    ):
        self.n = 3 # 3 states (x, y, theta)
        self.m = 2 

        self._speed = speed
        self._goal_marker = None
        self._t = 0
        self._dt=dt
        self._last_v = np.array([0.0,0.0])
        self.horizon = horizon

        N = 50
        traj = np.zeros((N, 3))

        #self.Q = np.array([[0.1,0.0,0.0],
        #             [0.0,0.1,0.0],
        #             [0.0,0.0,0.0]])

        self.Q = np.array([[0.1,0.0],
                     [0.0,0.1]])

        self.R = np.array([[0.001, 0.0],
                      [0.0, 0.001]])

        self.u_ini = np.zeros(horizon*2)

        self.uMax = np.array([1.0,1.0])


        self._seed = seed
        self.verbose = True
        self._veh = vehicle
        self.inputs = []
        self.states = []
        self._W = 1.0
        self.relativeState = relativeState

    def init(self, initObs):
        pass


    # Solve MPC optimization problem
    def solve_mpc(self, x0, traj):
        # Bounds --> -tau_max <= tau[idx] <= tau_max for idx = 0 to N-1
        bounds = [(-self.uMax[idx%2], self.uMax[idx%2]) for idx in range(self.horizon*2)]
 
        # Starting optimisation point for theta and dtheta are the current measurements
 
        # Minimization
        result = minimize(mpc_cost, self.u_ini, args=(self.u_ini,x0, traj, self.Q, self.R, self.horizon), bounds=bounds,method='SLSQP')
 
        # Extract the optimal control sequence
        u_mpc = two_dim_from_flat(result.x, 2)
 
        return u_mpc

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
        #traj = [t.numpy() for t in traj]
        if self.relativeState:
            s = np.array([0.0,0.0,0.0])

        us = self.solve_mpc(s, traj)
        # Save for warmstart on next minimization
        self.u_ini = to_flat(us)

        v = us[0]

        v += np.random.multivariate_normal(np.zeros(2), noise)
        self.inputs.append(v)

        return self.make_control_callable(v)

    def make_control_callable(self, u):
        func = lambda obj, t, s : u
        return func

import numpy as np
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
def mpc_cost(u, u_ini, x0, traj, Q, R, N):
    cost = 0
    x = np.copy(x0)
    # Penalize input changes for smoothness
    prev_u = np.copy(u_ini[:2])
    for idx in range(0, N*2, 2):
        inp = np.array([u[idx], u[idx+1]])
        x = unicycle_step(x, inp)
        # Tracking cost (position)
        step_idx = idx // 2
        cost += (x[:2] - traj[step_idx][:2]).T @ Q @ (x[:2] - traj[step_idx][:2])
        # Heading cost (optional, penalize angle error)
        th_err = np.arctan2(np.sin(x[2] - traj[step_idx][2]), np.cos(x[2] - traj[step_idx][2]))
        cost += 0.5 * th_err**2
        # Input cost
        cost += inp.T @ R @ inp
        # Input change penalty
        cost += 0.5 * np.sum((inp - prev_u)**2)
        prev_u = inp
    # Terminal cost (strong penalty for final position and heading)
    cost += 5.0 * (x[:2] - traj[-1][:2]).T @ Q @ (x[:2] - traj[-1][:2])
    th_err = np.arctan2(np.sin(x[2] - traj[-1][2]), np.cos(x[2] - traj[-1][2]))
    cost += 1.0 * th_err**2
    return cost

class SimpleMPC(ControllerWrapper):
    def __init__(
        self,
        speed=1,
        seed=0,
        dt=0.1,
        horizon=12,
        vehicle=None,
        relativeState=True
    ):
        self.n = 3 # 3 states (x, y, theta)
        self.m = 2 
        self._speed = speed
        self._goal_marker = None
        self._t = 0
        self._dt=dt
        self._last_v = np.array([0.0,0.0])
        self.horizon = horizon
        self.Q = np.array([[0.6,0.0],[0.0,0.6]])
        self.R = np.array([[0.1, 0.0],[0.0, 0.1]])
        self.u_ini = np.zeros(horizon*2)
        self.uMax = np.array([3.0,3.0])
        self._seed = seed
        self.verbose = True
        self._veh = vehicle
        self.inputs = []
        self.states = []
        self._W = 1.0
        self.relativeState = relativeState

    def solve_mpc(self, x0, traj):
        # Pad trajectory if needed
        traj_pad = np.zeros((self.horizon, 3))
        for i in range(min(self.horizon, len(traj))):
            traj_pad[i,:] = traj[i]
        bounds = [(-self.uMax[idx%2], self.uMax[idx%2]) for idx in range(self.horizon*2)]
        result = minimize(mpc_cost, self.u_ini, args=(self.u_ini, x0, traj_pad, self.Q, self.R, self.horizon), bounds=bounds, method='SLSQP')
        u_mpc = two_dim_from_flat(result.x, 2)
        return u_mpc

    def demand(self, s, traj, noise):
        speed = self._speed
        if self.relativeState:
            s = np.array([0.0,0.0,0.0])
        us = self.solve_mpc(s, traj)
        self.u_ini = to_flat(us)
        v = us[0]
        self.inputs.append(v)
        return self.make_control_callable(v)

    def make_control_callable(self, u):
        func = lambda obj, t, s : u
        return func

import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
import casadi as ca
import math
from controllers.pid_stub import PIDStub
from common.se2 import SE2
from common.transformations import global2LocalCoords
from roboticstoolbox.mobile.drivers import VehicleDriverBase
from spatialmath import base


def get_coeffs(t_mat):
    return np.array([t_mat[0,2], t_mat[1,2], t_mat[1,0]])

# Adaptation of the Geometric MPC to run as a driver in the pyrobotics toolbox
# Used for datageneration
class GenGeometricMPC(VehicleDriverBase):
    def __init__(self,
                 workspace,
                 trajectory,
                 speef=2,
                 seed=0,
                 dt=0.1,
                 vehicle=None
                 ):

        if hasattr(workspace, "workspace"):
            # workspace can be defined by an object with a workspace attribute
            self._workspace = base.expand_dims(workspace.workspace)
        else:
            self._workspace = base.expand_dims(workspace)
        self.nState = 3  # twist error (se2 vee) R^3
        self.nControl = 2  # velocity control (v, w) R^2
        self.nTwist = 3  # twist (se2 vee) R^3
        self.nTraj = None
        self.ref_state = None
        self.ref_control = None
        self.dt = None
        self.Q = None
        self.R = None
        self.N = None
        self.solve_time = 0.0
        self.setup_solver()
        self.set_control_bound()
        self.dt=dt
        self.ref_controller = PIDStub()
        # This trajectory object can be sampled for states and is distinct from the 
        # reference states
        self._traj_obj = trajectory

        self._t = 0

        self._workspace=workspace

        self._veh = vehicle

        self.inputs =[]
        self.states = []

    def init(self, ax=None):
        pass

    def set_traj(self, t):
        # Pull the trajectory out of the object from the current state to the end of our horizon
        self.ref_state = np.array([self._traj_obj.eval(t_c*self.dt) for t_c in range(t, t+self.N+1)])
        self.ref_state = self.ref_state.T
        self.nTraj = self.ref_state.shape[1]
        x = self.ref_state[:,0].T
        noise = np.zeros((2,2))
        controls = []
        for i in range(self.nTraj):

            u = self.ref_controller.demand(x, self.ref_state[:,i:].T, noise)(None,None,None)
            controls.append(u)
            x = self.ref_state[:,i].T

        self.ref_control = np.array(controls).T

    def setup_solver(self, Q=[20000, 20000, 200], R=0.3, N=10):
        self.Q = np.diag(Q)
        self.R = R * np.diag(np.ones(self.nControl))
        self.N = N

    def set_control_bound(self, v_min = -3, v_max= 3, w_min = -3, w_max= 3):
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max


    def demand(self):
        """
        current_state: current state of the system (x, y, theta)
        t: time -> index of reference trajectory (t = k * dt)

        return:
            u: control input (v, w)
        """

        self.set_traj(self._t)

        start_time = time.time()

        # get reference state and twist
        curr_ref = self.ref_state[:, 0]
        current_state = self._veh._x + np.random.normal(loc=0.0, scale=0.02, size=(3))
        self.states.append(current_state)

        # get x init by calculating log between current state and reference state
        lie_state = SE2(vec=current_state).t_matrix
        lie_des = SE2(vec=curr_ref).t_matrix.double()
        x_init = get_coeffs(np.squeeze(scipy.linalg.logm(lie_des.inverse() @ lie_state)))
        
        Q = self.Q
        R = self.R
        N = self.N
        dt = self.dt

        # setup casadi solver
        opti = ca.Opti('conic')
        # opti = ca.Opti()
        x_var = opti.variable(self.nState, N + 1)
        u_var = opti.variable(2, N)

        # setup initial condition
        opti.subject_to(x_var[:, 0] == x_init)

        # setup dynamics constraints
        # x_next = A * x + B * u + h
        for i in range(N):
            index = min(i, self.nTraj - 1)
            u_d = self.ref_control[:, index]  # desir
            u_d = self.vel_cmd_to_local_twist(u_d)
            A = -self.adjoint(self.carat(np.squeeze(u_d)))
            B = np.eye(self.nTwist)
            h = -u_d
            x_next = x_var[:, i] + dt * (A @ x_var[:, i] + B @ self.vel_cmd_to_local_twist(u_var[:, i]) + h)
            opti.subject_to(x_var[:, i + 1] == x_next)

        # cost function
        cost = 0
        for i in range(N):
            index = min(i, self.nTraj - 1)
            u_d = self.ref_control[:, index]
            cost += ca.mtimes([x_var[:, i].T, Q, x_var[:, i]]) + ca.mtimes(
                [(u_var[:, i]-u_d).T, R, (u_var[:, i]-u_d)])

        cost += ca.mtimes([x_var[:, N].T, 100*Q, x_var[:, N]])

        # control bound
        opti.subject_to(u_var[0, :] >= self.v_min)
        opti.subject_to(u_var[0, :] <= self.v_max)
        opti.subject_to(u_var[1, :] >= self.w_min)
        opti.subject_to(u_var[1, :] <= self.w_max)


        opts_setting = { 'printLevel': 'none'}
        opti.minimize(cost)
        opti.solver('qpoases',opts_setting)
        sol = opti.solve()
        psi_sol = sol.value(x_var)
        u_sol = sol.value(u_var)
        end_time = time.time()
        self.solve_time = end_time - start_time

        self._t += 1

        self.inputs.append(u_sol[:, 0])

        return u_sol[:, 0]

    def get_solve_time(self):
        return self.solve_time

    def vel_cmd_to_local_twist(self, vel_cmd):
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def local_twist_to_vel_cmd(self, local_vel):
        return ca.vertcat(local_vel[0], local_vel[2])

    @staticmethod
    def adjoint(xi):
        """Takes adjoint of element in SE(3)

        Args:
            xi (3x3 ndarray) : Element in Lie Group

        Returns:
            Ad_xi (3,3 ndarray) : Adjoint in SE(3)"""
        # make the swap
        xi[0,2], xi[1,2] = xi[1,2], -xi[0,2]
        return xi

    @staticmethod
    def carat(xi):
        """Moves an vector to the Lie Algebra se(3).

        Args:
            xi (3 ndarray) : Parametrization of Lie algebra

        Returns:
            xi^ (3,3 ndarray) : Element in Lie Algebra se(2)"""
        return np.array([[0,   -xi[2], xi[0]],
                        [xi[2], 0,     xi[1]],
                        [0,     0,     0]])

    def make_control_callable(self, u):
        func = lambda obj, t, s : u
        return func

    def reset(self):
        self.states = []
        self.inputs = []

    def get_run_stats(self):
        return self.states, self.inputs



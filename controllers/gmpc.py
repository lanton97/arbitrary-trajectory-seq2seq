import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
import casadi as ca
import math
from controllers.pid_stub import PIDStub
from common.se2 import SE2
from common.transformations import global2LocalCoords

"""
this GeometricMPC class is used to solve tracking problem of uni-cycle model
using MPC. The error dynamics is defined as follows:
error dynamics:
    psi_dot = At * psi_t + Bt * ut + ht
state:
    psi: lie algebra element of Psi (SE2 error)
control:
    ut = xi_t: twist (se2 element)

State transition matrix:
    At: ad_{xi_d,t}
Control matrix:
    B_k = I
offset:
    ht = xi_t,d: desired twist (se2 element)
    
the reference trajectory is generated using TrajGenerator class in ref_traj_generator.py
"""

def get_coeffs(t_mat):
    return np.array([t_mat[0,2], t_mat[1,2], t_mat[1,0]])

def get_coeffs2(t_mat):
    return np.array([t_mat[0,2], t_mat[1,2], np.arctan2(t_mat[1,0], t_mat[0,0])])

class GeometricMPC:
    def __init__(self, dt=0.02,relativeState=False):
        self.nState = 3  # twist error (se2 vee) R^3
        self.nControl = 2  # velocity control (v, w) R^2
        self.nTwist = 3  # twist (se2 vee) R^3
        self.relativeState = relativeState
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
        self.last_u = np.zeros(self.nControl)

    def set_traj(self, traj):
        self.ref_state = traj.T
        self.nTraj = self.ref_state.shape[1]
        x = self.ref_state[:,0].T
        noise = np.zeros((2,2))
        controls = []
        for i in range(self.nTraj):

            u = self.ref_controller.demand(x, self.ref_state[:,i:].T, noise)(None,None,None)
            controls.append(u)
            x = self.ref_state[:,i].T

        self.ref_control = np.array(controls).T

    def setup_solver(self, Q=[20000, 20000, 200], R=0.3, N=20, F=10):
        self.Q = np.diag(Q)
        self.R = R * np.diag(np.ones(self.nControl))
        self.N = N
        self.F=F

    def set_control_bound(self, v_min = -2.0, v_max= 2.0, w_min = -2.0, w_max= 2.0, dv_max=0.8, dw_max=0.8):
        self.v_min = 0.5#v_min
        self.v_max = 0.5#v_max
        self.w_min = w_min
        self.w_max = w_max
        self.dv_max = dv_max
        self.dw_max = dw_max


    def demand(self, current_state, traj, noise):
        """
        current_state: current state of the system (x, y, theta)
        t: time -> index of reference trajectory (t = k * dt)

        return:
            u: control input (v, w)
        """

        self.set_traj(traj)

        start_time = time.time()
        if self.ref_state is None:
            raise ValueError('Reference trajectory is not set up yet!')

        # get reference state and twist
        curr_ref = self.ref_state[:, 0]

        # get x init by calculating log between current state and reference state
        lie_state = SE2(vec=current_state).t_matrix
        if curr_ref.shape[0]==3:
            lie_des = SE2(vec=curr_ref).t_matrix.double()
        else:
            lie_des = SE2(man_vec=curr_ref).t_matrix.double()

        # If we are using a relative state, our desired traj is directly the negative error
        if self.relativeState:
            x_init = -get_coeffs2(lie_des)
        else:
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

        cost += ca.mtimes([x_var[:, N].T, self.F*Q, x_var[:, N]])

        # control bound
        opti.subject_to(u_var[0, :] >= self.v_min)
        opti.subject_to(u_var[0, :] <= self.v_max)
        opti.subject_to(u_var[1, :] >= self.w_min)
        opti.subject_to(u_var[1, :] <= self.w_max)

        # Acceleration max
        opti.subject_to(ca.fabs(u_var[0,:] - self.last_u[0]) <= self.dv_max)
        opti.subject_to(ca.fabs(u_var[1,:] - self.last_u[1]) <= self.dv_max)

        opts_setting = { 'printLevel': 'none'}
        opti.minimize(cost)
        opti.solver('qpoases',opts_setting)
        sol = opti.solve()
        psi_sol = sol.value(x_var)
        u_sol = sol.value(u_var)
        end_time = time.time()
        self.solve_time = end_time - start_time
        self.last_u = u_sol[:,0]
        return self.make_control_callable(u_sol[:, 0])

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

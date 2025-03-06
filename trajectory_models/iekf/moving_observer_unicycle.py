import numpy as np
from scipy.linalg import expm
from roboticstoolbox.mobile import Unicycle
from common.se2 import *
from common.transformations import global2LocalCoords
from .unicycle_wrapper import *

class MovingObserverUnicycle(UnicycleWrapper):

    def __init__(self, Q, R, deltaT):
        """The basic unicycle model. 

        Args:
            Q (3,3 nparray): Covariance of noise on state
            R (2x2 nparray): Covariance of noise on measurements"""
        self.Q = Q
        self.R = R
        self.deltaT = deltaT
        self.b = np.array([0, 0, 1])
        self.unicycle = Unicycle()

    def gen_data(self, ego_x0, observed_x0, ego_u, observed_u, t, noise=True):
        """Generates model data using Lie Group model.

        Args:
            x0 (3,3 ndarray or 3 ndarray) : Starting point of model
            u        (basically anything) : Control to apply to state. Can be function, integer, or 2 ndarray.
            t                       (int) : How many timesteps to run
            noise                  (bool) : Whether or not to include noise. Defaults to True.

        Returns:
            x (t,3,3 ndarray) : x steps after x0 
            u   (t,3 ndarray) : controls that were applied
            z   (t,3 ndarray) : measurements taken.
        """
        #Accept various forms of u. Parse them here
        #as a function of t
        if hasattr(ego_u, '__call__'):
            ego_u = np.array([ego_u(t) for t in range(t)])
        #as an int, in which case it'll default to that int everywhere
        elif isinstance(ego_u, int):
            ego_u = np.zeros((t, 2)) + ego_u
        elif len(ego_u.shape) == 1:
            ego_u = np.tile(ego_u, (t,1))

        if hasattr(observed_u, '__call__'):
            observed_u = np.array([observed_u(t) for t in range(t)])
        #as an int, in which case it'll default to that int everywhere
        elif isinstance(observed_u, int):
            observed_u = np.zeros((t, 2)) + observed_u
        elif len(observed_u.shape) == 1:
            observed_u = np.tile(observed_u, (t,1))

        ego_x = np.zeros((t+1,3, 3))
        obs_x = np.zeros((t+1,3, 3))
        ego_z = np.zeros((t+1, 3))
        obs_z = np.zeros((t+1, 3))

        true_obs_x = np.zeros((t+1,3, 3))

        # convert u -> algebra -> group
        if ego_u.shape != (t,2) and ego_u.shape != (t,2):
            raise ValueError("Wrong Sized Shape for control!")
        if observed_u.shape != (t,2) and observed_u.shape != (t,2):
            raise ValueError("Wrong Sized Shape for control!")

        #convert x0 into lie group if needed
        if observed_x0.shape == (3,):
            observed_x0 = np.array([[np.cos(observed_x0[2]), -np.sin(observed_x0[2]), observed_x0[0]],
                            [np.sin(observed_x0[2]), np.cos(observed_x0[2]), observed_x0[1]],
                            [0,             0,             1]])
        elif observed_x0.shape != (3,3):
            raise ValueError("Wrong Sized Shape for x0!")

        if ego_x0.shape == (3,):
            ego_x0 = np.array([[np.cos(ego_x0[2]), -np.sin(ego_x0[2]), ego_x0[0]],
                            [np.sin(ego_x0[2]), np.cos(ego_x0[2]), ego_x0[1]],
                            [0,             0,             1]])
        elif ego_x0.shape != (3,3):
            raise ValueError("Wrong Sized Shape for x0!")

        ego_x[0] = ego_x0
        true_obs_x[0] = observed_x0
        obs_x[0] = self.transform(ego_x0, observed_x0)

        for i in range(1, t+1):
            ego_x[i] = self.f_lie(ego_x[i-1], ego_u[i-1], noise)
            ego_z[i] = self.h(ego_x[i], noise)

            true_obs_x[i] = self.f_lie(true_obs_x[i-1], observed_u[i-1], noise)

            obs_x[i] = self.f_lie_leader(obs_x[i-1], observed_u[i-1],  ego_x[i-1], ego_x[i], noise) 

            obs_z[i] = self.h(obs_x[i], noise)

        return ego_x[1:], ego_z[1:], obs_x[1:], obs_z[1:], true_obs_x[1:]

    
    def f_lie(self, state, u, noise=False):
        """Propagates state forward in Lie Group. Used for gen_data and IEKF.

        Args:
            state (,3 ndarray) : X_n of model in Lie Group
            u     (3 ndarray) : U_n of model as a vector
            noise        (bool) : Whether or not to add noise. Defaults to False.

        Returns:
            X_{n+1} (3,3 ndarray)"""
        if noise:
            w = np.random.multivariate_normal(mean=np.zeros(3), cov=self.Q)
        else:
            w = np.zeros(3)
        return state @ expm(carat( np.array([u[0], 0, u[1]] + w )*self.deltaT ))

    def f_lie_leader(self, state, u, ego_x0, ego_x1, noise=False):
        if noise:
            w = np.random.multivariate_normal(mean=np.zeros(3), cov=self.Q)
        else:
            w = np.zeros(3)

        nstate = self.inverseTransform(ego_x0, state)

        return self.transform(ego_x1, nstate @ expm(self.carat( np.array([u[0], 0, u[1]] + w )*self.deltaT )))

    def f_standard(self, state, u, noise=False):
        """Propagates state forward in regular coordinates. Used for EKF.

        Args:
            state (3 ndarray): X_n of model in coordinates
            u     (3 ndarray): U_n of model in standard coordinates
            noise     (bool) : Whether or not to add noise. Defaults to False.

        Returns:
            X_{n+1} (3 ndarray)"""

        if noise:
            w = np.random.multivariate_normal(mean=np.zeros(3), cov=self.Q)
        else:
            w = np.zeros(3)
        u = (u.copy() + w[[0,2]])*self.deltaT
        new_state = self.unicycle.f(state, u)
        return new_state

    def h(self, state, noise=False):
        """Calculates measurement given a state. Note that the result is
            the same if it's in standard or Lie Group form, so we simplify into
            one function.
            
        Args:
            state (3 ndarray or 3,3 ndarray) : Current state in either standard or Lie Group form
            noise                     (bool) : Whether or not to add noise. Defaults to False.

            
        Returns:
            Z_n (3 ndarray or 3,3 ndarray)"""
        # using standard coordinates
        if state.shape == (3,):
            z = np.array([state[0], state[1], state[2]])
        # using Lie Group
        elif state.shape == (3,3):
            z = state @ self.b

        #add noise if needed
        if noise:
            w = np.random.multivariate_normal(mean=np.zeros(2), cov=self.R)
            z[:2] += w

        return z

    def F(self, state, u):
        """Jacobian of system using standard coordinates. Used for EKF.

         Args:
            state (3 ndarray) : X_n of model in coordinates
            u     (3 ndarray) : U_n of model in standard coordinates

        Returns:
            df / dx (3,3 ndarray)"""
        return self.unicycle.Fx(state, u)

    def F_u(self, state, u):
        return self.unicycle.Fv(state, u)

    def H(self, state):
        """Jacobian of measurement model using standard coordinates. Used for EKF.

         Args:
            state (3 ndarray) : X_n of model in coordinates
            u     (3 ndarray) : U_n of model in standard coordinates

        Returns:
            dh / dx (3,3 ndarray)"""
        return np.array([[1, 0, 0],
                         [0, 1, 0]])



# we do our testing down here
if __name__ == "__main__":
    # setup system
    Q = np.diag([.001, 0, .1])
    R = np.diag([.001, .001])
    dt = 0.1
    sys = MovingObserverUnicycle(Q, R, dt)

    x0 = np.array([0, 0, 0])
    obs_x0 = np.array([0, 0, 0])# np.array([-5, -5, np.pi/2])

    # generate data from Lie Group method
    t = 100
    u = lambda t: np.array([1, np.sin(t/2)])
    u2 = lambda t: np.array([1, np.sin(t/2)])
    xl, zl, obs_x, obs_z, true_obs_x = sys.gen_data(obs_x0, x0, u, u2, t, noise=True)

    # generate data from standard method
    us = np.array([u(t) for t in range(t)])

    def translateToCoordinateBase(refTStep, baseTraj, observedTraj):
        ref_point = baseTraj[refTStep]
        refViews = []
        baseViews = []
        zeroView = sys.vecToSE2([0, 0, 0])
        for base_point, observed_point  in zip(baseTraj, observedTraj):
            # translate
            baseView = sys.inverseTransform(base_point, observed_point)
            refView = sys.transform(ref_point, baseView)
            refViews.append(refView)
            baseViews.append(baseView)

        return np.array(refViews), np.array(baseViews)

    base, ref_traj = translateToCoordinateBase(-1, xl, obs_x)

    #plot data
    import matplotlib.pyplot as plt
    plt.plot(xl[:,0,2], xl[:,1,2], label="Lie Group Method (Noisy)")
    plt.plot(zl[:,0], zl[:,1], label="Lie Group Measurements", alpha=0.5)
    #plt.plot(obs_x[:,0,2], obs_x[:,1,2], label="Relative Lie Group Method (Noisy)")
    #plt.plot(obs_z[:,0], obs_z[:,1], label="Relative Lie Group Measurements", alpha=0.5)
    plt.plot(true_obs_x[:,0,2], true_obs_x[:,1,2], label="True Secondary Lie Group Method (Noisy)")
    plt.plot(ref_traj[:,0,2], ref_traj[:,1,2], label="reference trajectory")
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(obs_z[:,0], label="X Distances", alpha=0.5)
    plt.plot(obs_z[:,1], label="Y Distances", alpha=0.5)
    plt.legend()
    plt.show()


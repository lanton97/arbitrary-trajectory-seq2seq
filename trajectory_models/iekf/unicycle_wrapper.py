import numpy as np
from scipy.linalg import expm
from roboticstoolbox.mobile import Unicycle
from common.se2 import *
from common.util import *
from common.transformations import *

class UnicycleWrapper:

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

    def gen_data(self, x0, u, t, noise=True):
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
        if hasattr(u, '__call__'):
            u = np.array([u(t) for t in range(t)])
        #as an int, in which case it'll default to that int everywhere
        elif isinstance(u, int):
            u = np.zeros((t, 2)) + u
        elif len(u.shape) == 1:
            u = np.tile(u, (t,1))

        x = np.zeros((t+1,3, 3))
        x_lie = np.zeros((t+1, 3, 3))
        z = np.zeros((t+1, 3))

        # convert u -> algebra -> group
        if u.shape != (t,2):
            raise ValueError("Wrong Sized Shape for control!")

        #convert x0 into lie group if needed
        if x0.shape == (3,):
            x0 = np.array([[np.cos(x0[2]), -np.sin(x0[2]), x0[0]],
                            [np.sin(x0[2]), np.cos(x0[2]), x0[1]],
                            [0,             0,             1]])
        elif x0.shape != (3,3):
            raise ValueError("Wrong Sized Shape for x0!")

        x[0] = x0

        for i in range(1, t+1):
            x[i] = self.f_lie(x[i-1], u[i-1], noise)
            z[i] = self.h(x[i], noise)

        return x[1:], u, z[1:]

    
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
        return state @ expm(self.carat( np.array([u[0], 0, u[1]] + w )*self.deltaT ))

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

    # translate obs x into the base_x coordinate frame, as if
    # origin of the frame is at basex
    @staticmethod
    def transform(base_x, obs_x):
        base_vec = np.array([base_x[0,2], base_x[1,2], np.arctan2(base_x[1,0], base_x[0,0])])
        observed_vec = np.array([obs_x[0,2], obs_x[1,2], np.arctan2(obs_x[1,0], obs_x[0,0])])
        translated_vec = global2LocalCoords(base_vec, observed_vec)

        return UnicycleWrapper.vecToSE2(translated_vec)

    @staticmethod
    def inverseTransform(base, observed):
        translation = UnicycleWrapper.vecToSE2(-np.array([base[0,2], base[1,2], 0]))
        rotation = UnicycleWrapper.vecToSE2(np.array([0, 0, -np.arctan2(base[1,0], base[0,0])]))
        view = UnicycleWrapper.transform(rotation, observed)
        view = UnicycleWrapper.transform(translation, view)
        return view

    # Convert vector to se2 matrix
    @staticmethod
    def vecToSE2(xi):
        return np.array([[np.cos(xi[2]), -np.sin(xi[2]), xi[0]],
                        [np.sin(xi[2]), np.cos(xi[2]), xi[1]],
                        [0,             0,             1]])

# we do our testing down here
if __name__ == "__main__":
    # setup system
    Q = np.diag([.001, 0, .1])
    R = np.diag([.001, .001])
    dt = 0.1
    sys = UnicycleWrapper(Q, R, dt)
    x0 = np.zeros(3)

    # generate data from Lie Group method
    t = 100
    u = lambda t: np.array([1, np.sin(t/2)])
    xl, ul, zl = sys.gen_data(x0, u, t, noise=True)

    # generate data from standard method
    us = np.array([u(t) for t in range(t)])
    xs = np.zeros((t+1, 3))
    xs[0] = x0
    for i in range(1,t+1):
        xs[i] = sys.f_standard(xs[i-1], us[i-1], noise=False)

    #plot data
    import matplotlib.pyplot as plt
    plt.plot(xl[:,0,2], xl[:,1,2], label="Lie Group Method (Noisy)")
    plt.plot(zl[:,0], zl[:,1], label="Lie Group Measurements", alpha=0.5)
    plt.plot(xs[:,0], xs[:,1], label="Coordinate Based Method (No Noise)")
    plt.legend()
    plt.show()


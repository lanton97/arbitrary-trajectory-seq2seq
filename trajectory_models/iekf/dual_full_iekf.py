import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm
from .iekf import InvariantEKF
import matplotlib.pyplot as plt
from common.se2 import adjoint, carat

class movingWaypointIEKF(InvariantEKF):
    def __init__(self, sys, mu0, sigma0):
        super(movingWaypointIEKF, self).__init__(sys, mu0, sigma0)
        self.Ks = []

    def predict(self, u, ego_x0, ego_x1):
        """Runs prediction step of iEKF.

        Args:
            u       (k ndarray) : control taken at this step

        Returns:
            mu    (nxn ndarray) : Propagated state
            sigma (nxn ndarray) : Propagated covariances"""
        
        #get mubar and sigmabar
        mu_bar = self.sys.f_lie_leader(self.mu, u, ego_x0, ego_x1)
        adj_u = adjoint( inv(expm(carat( np.array([u[0], 0, u[1]])*self.sys.deltaT ))) )
        sigma_bar = adj_u @ self.sigma @ adj_u.T + self.sys.Q * self.sys.deltaT**2

        #save for use later
        self.mus.append( mu_bar )
        self.sigmas.append( sigma_bar )

        return mu_bar, sigma_bar

    def update(self, z):
        """Runs correction step of iEKF.

        Args:
            z (m ndarray): measurement at this step

        Returns:
            mu    (nxn ndarray) : Corrected state
            sigma (nxn ndarray) : Corrected covariances"""

        H = np.array([[1, 0, 0],
                      [0, 1, 0]])
        V = ( inv( self.mu )@z - self.sys.b )[:-1]

        invmu = inv(self.mu)[:2,:2]
        K = self.sigma @ H.T @ inv(H@self.sigma@H.T + invmu@self.sys.R@invmu.T )
        self.Ks.append(K)
        self.mus[-1] = self.mu @ expm( carat(K @ V) )
        self.sigmas[-1] = (np.eye(3) - K @ H) @ self.sigma

        return self.mu, self.sigma


class DualIEKFTracker:
    def __init__(self, sys, ego_mu0, ego_sigma0, leader_mu0, leader_sigma0):
        self.system = sys
        self.egoTracker = InvariantEKF(sys, ego_mu0, ego_sigma0)
        self.leaderTracker = movingWaypointIEKF(sys, leader_mu0, leader_sigma0)

    def iterate(self, ego_u, ego_z, leader_u, leader_z):
        for eu, ez, lu, lz in zip(ego_u, ego_z, leader_u, leader_z):
            self.egoTracker.predict(eu)
            self.egoTracker.update(ez)
            self.leaderTracker.predict(lu, self.egoTracker.mus[-2], self.egoTracker.mu)
            self.leaderTracker.update(lz)
        return np.array(self.egoTracker.mus)[1:], np.array(self.egoTracker.sigmas)[1:], \
                np.array(self.leaderTracker.mus)[1:], np.array(self.leaderTracker.sigmas)[1:]

    def step(self, ego_u, ego_z, leader_u, leader_z):
        self.egoTracker.predict(ego_u)
        self.egoTracker.update(ego_z)
        self.leaderTracker.predict(leader_u, self.egoTracker.mus[-2], self.egoTracker.mu)
        self.leaderTracker.update(leader_z)
        return np.array(self.egoTracker.mus)[-100:], np.array(self.egoTracker.sigmas)[-100:], \
                np.array(self.leaderTracker.mus)[-100:], np.array(self.leaderTracker.sigmas)[-100:]

    def translateToCoordinateBase(self, refTStep, baseTraj, observedTraj):
        ref_point = baseTraj[refTStep]
        refViews = []
        baseViews = []
        zeroView = self.system.vecToSE2([0, 0, 0])
        for base_point, observed_point  in zip(baseTraj, observedTraj):
            # translate
            #refView = self.translateSinglePointToCoordinateBase(baseTraj, observed_point)
            baseView = self.system.inverseTransform(base_point, observed_point)
            refView = self.system.transform(ref_point, baseView)
            refViews.append(refView)
            baseViews.append(baseView)

        return np.array(refViews), np.array(baseViews)

    def translateSinglePointToCoordinateBase(self, basePoint, refPoint, observedPoint):
        # Transform the observation to a global reference frame
        baseView = self.system.inverseTransform(basePoint, observedPoint)
        # Transform from the global frame to the new relative frame
        refView = self.system.transform(refPoint, baseView)

        return refView

if __name__ == "__main__":
    from .moving_observer_unicycle import MovingObserverUnicycle

    # setup system
    Q = np.diag([.001, 0, .1])
    R = np.diag([.001, .001])
    dt = 0.1
    sys = MovingObserverUnicycle(Q, R, dt)
    ego_x0 = np.zeros(3)
    leader_x0 = np.array([2.0, 1.0, 0])

    # generate data from Lie Group method
    t = 100
    u = lambda t: np.array([1, np.sin(t/2)])
    xe, ze, xl, zl, true_xl = sys.gen_data(ego_x0, leader_x0, u, u, t, noise=True)

    # Run the iekf
    u = np.array([u(t) for t in range(t)])
    iekf = DualIEKFTracker(sys, ego_x0, np.eye(3), leader_x0, np.eye(3))
    emus, esigmas, lmus, lsigmas = iekf.iterate(u, ze, u, zl)

    relative_traj, base_traj = iekf.translateToCoordinateBase(-1, emus, lmus)
    true_relative_traj, _ = iekf.translateToCoordinateBase(-1, xe, xl)

    # plot results
    plt.clf()

    plt.plot(relative_traj[:,0,2], relative_traj[:,1,2], label="Estimated Relative Trajectory")
    plt.plot(true_relative_traj[:,0,2], true_relative_traj[:,1,2], label="True Relative Location")
    #plt.plot(base_traj[:,0,2], base_traj[:,1,2], label="Estimated Base Trajectory")
    plt.plot(xe[:,0,2], xe[:,1,2], label="Actual Ego Location")
    plt.plot(true_xl[:,0,2], true_xl[:,1,2], label="Actual Leader Location")
    plt.plot(emus[:,0,2], emus[:,1,2], label="iEKF Ego Results")
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(xl[:,0,2], label="X Distances", alpha=0.5)
    plt.plot(xl[:,1,2], label="Y Distances", alpha=0.5)
    plt.plot(lmus[:,0,2], label="IEKF X Distances", alpha=0.5)
    plt.plot(lmus[:,1,2], label="IEKF Y Distances", alpha=0.5)
    plt.legend()
    plt.show()

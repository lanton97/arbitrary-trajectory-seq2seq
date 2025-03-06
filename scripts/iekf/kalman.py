from trajectory_models.iekf.moving_observer_unicycle import *
from trajectory_models.iekf.dual_iekf import *
import matplotlib.pyplot as plt 

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

plt.plot(xe[:,0,2], xe[:,1,2], label="Actual Ego Location")
plt.plot(true_xl[:,0,2], true_xl[:,1,2], label="Actual Leader Location")
plt.plot(emus[:,0,2], emus[:,1,2], label="iEKF Ego Results")
plt.legend()
plt.show()

plt.plot(true_relative_traj[:,0,2], true_relative_traj[:,1,2], label="True Relative Leader Location")
plt.plot(relative_traj[:,0,2], relative_traj[:,1,2], label="iEKF Relative Leader Location")
plt.legend()
plt.show()

plt.clf()
plt.plot(xl[:,0,2], label="X Distances", alpha=0.5)
plt.plot(xl[:,1,2], label="Y Distances", alpha=0.5)
plt.plot(lmus[:,0,2], label="IEKF X Distances", alpha=0.5)
plt.plot(lmus[:,1,2], label="IEKF Y Distances", alpha=0.5)
plt.legend()
plt.show()

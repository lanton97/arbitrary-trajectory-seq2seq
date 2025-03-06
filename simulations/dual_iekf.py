import gym
import numpy as np
from roboticstoolbox.mobile import Unicycle,PolygonMap, RRTPlanner,VehiclePolygon
from controllers.datagen.controller import *
from spatialmath import Polygon2, Ellipse
from roboticstoolbox.tools.trajectory import *
import matplotlib.pyplot as plt
from klampt.model.trajectory import path_to_trajectory
from klampt.model import trajectory
from common.transformations import global2LocalCoords
from copy import deepcopy
from controllers.pid_stub import *
from controllers.gmpc import *
from trajectory_models.iekf.iekf import *
from trajectory_models.iekf.dual_iekf import *
from trajectory_models.iekf.moving_observer_unicycle import *
import signal

# Runs IEKF for a convoy to evaluate accuracy of leader vehicle trajectory
class dual_iekf_convoy(gym.Env):

    # Create a 50 x 50 unit space for reach avoid games
    def __init__(self, maxSteps=3000, leader_inps=True, control=PIDStub):

        # Magnitude of the bounds
        self.bound_x = 100.0
        self.bound_y = 100.0
        self.goal_pos = None
        # Settings for the obstacles and size 
        self.num_obstacles = 20
        self.mean_obst_rad = 7
        self.obst_std = 1.414
        self.obsts = None
        # Timesteps ahead we record the trajectory
        self.offset = 100
        self.dt = 0.1
        self.max_length = maxSteps

        self.b = np.array([0,0,1])


        # Our controller and trajectory tracker
        self.controller = control(relativeState=True)
        #self.controller = GeometricMPC(relativeState=True)

        self.leaderController = PIDStub()

        # Default args for bicycle kinematics for now
        w=1
        # Bounding box for rrt collision checking
        self.vpolygon = Polygon2([(-w/2, w/2), (-w/2, -w/2), (w/2, -w/2), (w/2, w/2)])
        self.vehicle = Unicycle(dt=0.1, polygon=self.vpolygon)

    # Run the simulations and return the givven inputs
    def simulate(self, debug=False, plot=False):

        initInput = np.array([0.0,0.0])
        inputs = [initInput]

        s = self.vehicle._x

        Q = np.diag([0.02,0.02,0.02])
        R = np.diag([0.02,0.02])
        leaderNoise = np.array([[0.0, 0.0],
                                [0.0,0.0]])
        # Initialize our IEKF

        sys = MovingObserverUnicycle(Q,R,self.dt)
        iekf = InvariantEKF(sys, s, np.eye(3))
        self.iekf = DualIEKFTracker(sys, s, np.eye(3),s, np.eye(3))

        # Initialize state tracking arrays
        states = []
        measurements = []
        inputs = []
        traj   = []
        est_states = []
        est_s = s
        leader_s = []
        leader_traj = []
        true_traj = []

        # Loop through for the maximum amount of time or until the goal is reached
        for i in range(self.max_length):
            # Get the trajectory state
            # Currently only one since we are using PID
            traj_s = [np.array(global2LocalCoords(s,self.traj.eval((i+j)*self.dt))) for j in range(100)]
            true_traj_s = self.traj.eval(i*self.dt)

            # Evaluate the IEKF 10 timesteps before
            ref_point = self.iekf.egoTracker.mu
            est_leader_traj    =  [manifoldToVector(self.iekf.translateSinglePointToCoordinateBase(eMu, ref_point, lMu)).numpy() for eMu, lMu in zip(self.iekf.egoTracker.mus[-100:],self.iekf.leaderTracker.mus[-100:])]
            # If we don't have enough for a real estimate, we use the fixed trajectory estimates
            if len(est_leader_traj)< self.offset:
                est_leader_traj = traj_s

            #est_leader_traj = np.array(est_leader_traj)
            leader_traj_s = self.traj.eval((i+self.offset)*self.dt)

            # Get our estimated trajectory from the tracker
            # Sample inputs
            u   = self.controller.demand(est_s, np.array(est_leader_traj), noise=R)
            leader_u = self.leaderController.demand(self.traj.eval((i+self.offset - 1)*self.dt),[self.traj.eval((i+self.offset)*self.dt)], leaderNoise)(None, None, None)

            # Step the vehicle model according to our input
            odo = self.vehicle.step(u)

            # Get the true vehicle state
            s = deepcopy(self.vehicle._x)
            # Copy true state before adding noise
            s[2] = wrap(s[2])
            states.append(np.array(s))

            leader_s = global2LocalCoords(s,self.traj.eval((i+self.offset)*self.dt), shiftToPiRange=False) + np.random.normal(0, 0.02, (3))
            # Add state noise
            s = s + np.random.normal(0, 0.02, (3))
            # Convert to GPS coordinates measurements
            z = self.iekf.system.vecToSE2(s) @ self.b
            leader_z = self.iekf.system.vecToSE2(leader_s) @ self.b

            # Get a numerical input for the IEKF
            u = u(None, None, None)

            # Estimate and iterate
            est_s, sig, est_l, l_sig = self.iekf.step(u, z, leader_u, leader_z)
            est_s = manifoldToVector(est_s[-1]).numpy()

            # Track the trajectory, inputs, etc.
            traj.append(np.array(traj_s))
            true_traj.append(np.array(true_traj_s))
            measurements.append(z)
            est_states.append(np.array(est_s))
            inputs.append(u)
            leader_traj.append(est_leader_traj)
            # Break if we get close to the goal position
            if np.linalg.norm(np.array(s) - self.goal_pos) <= 5.0 or np.linalg.norm(np.array(traj[-1])- self.goal_pos) <= 5.0:
                break

        # Plot the followed trajectory and estimated states
        if plot:
            states =  np.array(states)
            measurements = np.array(measurements)
            inputs = np.array(inputs)
            traj   = np.array(traj)
            est_states = np.array(est_states)
            true_traj = np.array(true_traj)

            plt.clf()
            plt.plot(true_traj[100:,0], true_traj[100:,1], label='Desired Trajectory')
            plt.plot(states[100:,0], states[100:,1], label='States')
            plt.legend()
            plt.show()
         
         
            #calculate leader trajectory true and estimated
            true_leader_traj = [self.traj.eval((i+j)*self.dt) for j in range(100)]
            true_leader_traj = np.array([global2LocalCoords(states[-1], true_pos) for true_pos in true_leader_traj])
         
            # Try calculating the leader trajectory using the true distances between the ego vehicle and leader
            # This will tell us if the problem is in the IEKF of the leader or the translation fuunction
            # First try with the iekf of the ego vehicle positions
            ref_point = self.iekf.egoTracker.mu
         
            est_leader_traj    =  [manifoldToVector(self.iekf.translateSinglePointToCoordinateBase(eMu, ref_point, lMu)).numpy() for eMu, lMu in zip(self.iekf.egoTracker.mus[-100:],self.iekf.leaderTracker.mus[-100:])]
            est_leader_traj = np.array(est_leader_traj)
         
         
            plt.clf()
            plt.plot(true_leader_traj[:,0], true_leader_traj[:,1], label='True Relative Leader Trajectory')
            plt.plot(est_leader_traj[:,0], est_leader_traj[:,1], label='Est Relative Leader Trajectory')
            plt.legend()
            plt.show()
         
            plt.clf()
            plt.plot(true_leader_traj[:,2], label='True Relative Leader Trajectory')
            plt.plot(est_leader_traj[:,2], label='Est. Relative Leader Trajectory')
            plt.legend()
            plt.show()

        ego_state_pkg = ([states], [est_states], [true_traj])
        leader_state_pkg = (leader_traj, traj)

        return ego_state_pkg, leader_state_pkg

    def reset(self):

        plt.cla()
        signal.signal(signal.SIGALRM, RRTTimeoutHandler) 
        signal.alarm(420)
        try:
           # Set random positions within the bounding space
           positions = (np.random.rand(2, 2) - 0.5) * (self.bound_x -5)*2
           thetas = np.random.rand(2,1) * 2.0 * np.pi 
           # Make sure the positions are far apart
           while np.linalg.norm(positions[0] - positions[1]) < 5.0 :
               positions = (np.random.rand(2, 2) - 0.5) * self.bound_x * 2.0
         
           self.start_pos = np.concatenate([positions[0], thetas[0]])
           self.goal_pos = np.concatenate([positions[1], thetas[1]])
         
         
           # Generate a map of obstacles for motion planning
           self.map = PolygonMap(workspace=[-self.bound_x,self.bound_x])
         
           obst_rads = np.random.normal(loc=self.mean_obst_rad, scale=self.obst_std, size=(self.num_obstacles))
           obst_positions = (np.random.rand(2, self.num_obstacles) - 0.5) * self.bound_x * 2.0
           for i in range(self.num_obstacles):
               Polygon = []
               # Create a circle from 32 points
               for j in range(4):
                   theta = 2*np.pi * j / 32
                   Polygon.append((obst_positions[0,i] + obst_rads[i]*np.cos(theta), obst_positions[1,i] + obst_rads[i]*np.sin(theta)))
               self.map.add(Polygon)
         
           # RRT planning for a path
           rrt = RRTPlanner(map=self.map, vehicle=self.vehicle, npoints=400)
           rrt.plan(goal=self.goal_pos, showsamples=True, showvalid=False)
           path, status = rrt.query(start=self.start_pos)
           # Create a trajectory from the path
           traj = trajectory.Trajectory(milestones=path.tolist())
           self.traj = path_to_trajectory(traj, speed=1, dt=0.1)
           # Reset the beginning of the path in case the rrt failed
           self.start_pos = path[0]
           # Initialize the vehicle start position
           self.vehicle.init(x0=self.start_pos)
        except Exception as exc:
            print(exc)
            self.reset()
        signal.alarm(0)

        # return the trajectory global view and the initial state
        return self.traj, self.start_pos

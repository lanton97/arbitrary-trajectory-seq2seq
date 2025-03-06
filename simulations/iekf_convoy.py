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
from trajectory_models.iekf.iekf import *
from trajectory_models.iekf.unicycle_wrapper import *

# Runs IEKF for a convoy to evaluate accuracy of ego vehicle tracking
class single_iekf_convoy(gym.Env):

    # Create a 50 x 50 unit space for reach avoid games
    def __init__(self, maxSteps=100, leader_inps=True):

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
        self.controller = PIDStub()

        # Default args for bicycle kinematics for now
        w=1
        # Bounding box for rrt collision checking
        self.vpolygon = Polygon2([(-w/2, w/2), (-w/2, -w/2), (w/2, -w/2), (w/2, w/2)])
        self.vehicle = Unicycle(dt=0.1, polygon=self.vpolygon)

    # Run the simulations and return the givven inputs
    def simulate(self, debug=False):

        initInput = np.array([0.0,0.0])
        inputs = [initInput]

        s = self.vehicle._x

        Q = np.diag([0.02,0.02,0.02])
        R = np.diag([0.02,0.02])
        # Initialize our IEKF

        sys = UnicycleWrapper(Q,R,self.dt)
        self.iekf = InvariantEKF(sys, s, np.eye(3))

        # Initialize state trackign arrays
        states = []
        measurements = []
        inputs = []
        traj   = []
        est_states = []
        est_s = s
        #u = lambda t: np.array([self.max_length/10, 1])
        #us = np.array([u(t) for t in range(self.max_length)])
        # Loop through for the maximum amount of time or until the goal is reached
        for i in range(self.max_length):
            # Get the trajectory state
            # Currently only one since we are using PID
            traj_s = self.traj.eval(i*self.dt)

            # Get our estimated trajectory from the tracker
            # Sample inputs
            u   = self.controller.demand(est_s, [traj_s], noise=R)#(None, None, None)

            #s = self.vehicle.f(s, u*0.1)

            # Step the vehicle model according to our input
            odo = self.vehicle.step(u)

            # Get the true vehicle state
            s = deepcopy(self.vehicle._x)
            # Copy true state before adding noise
            s[2] = wrap(s[2])
            states.append(s)

            # Add state noise
            s = s + np.random.normal(0, 0.02, (3))
            # Convert to GPS coordinates measurements
            z = self.iekf.sys.vecToSE2(s) @ self.b

            u = u(None, None, None)

            est_s, sig = self.iekf.step(u, z)
            est_s = manifoldToVector(est_s).numpy()
            #est_s[2] = est_s[2]
            

            # Track the trajectory, inputs, etc.
            traj.append(traj_s)
            measurements.append(z)
            est_states.append(est_s)
            inputs.append(u)
            # Break if we get close to the goal position
            if np.linalg.norm(np.array(s) - self.goal_pos) <= 5.0 or np.linalg.norm(np.array(traj[-1])- self.goal_pos) <= 5.0:
                break

        states =  np.array(states)
        measurements = np.array(measurements)
        inputs = np.array(inputs)
        traj   = np.array(traj)
        est_states = np.array(est_states)

        plt.clf()
        plt.plot(est_states[:,0], est_states[:,1], label='Estimated States')
        plt.plot(states[:,0], states[:,1], label='States')
        plt.legend()
        plt.show()

        plt.clf()
        plt.plot(est_states[:,2], label='Estimated States')
        plt.plot(states[:,2], label='States')
        plt.legend()
        plt.show()
        

        return np.array(states), np.array(est_states)

    def reset(self):
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

        # return the trajectory global view and the initial state
        return self.traj, self.start_pos

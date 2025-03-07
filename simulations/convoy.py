import gym
import numpy as np
from roboticstoolbox.mobile import Unicycle,PolygonMap, RRTPlanner,VehiclePolygon
from spatialmath import Polygon2, Ellipse
from roboticstoolbox.tools.trajectory import *
import matplotlib.pyplot as plt
from klampt.model.trajectory import path_to_trajectory
from klampt.model import trajectory
from common.transformations import global2LocalCoords, wrap
from copy import deepcopy
import signal
from controllers.pid_stub import *
from common.util import RRTTimeoutHandler
import random

class convoy(gym.Env):

    # Create a 50 x 50 unit space for reach avoid games
    def __init__(self, trajectory_tracker, controller, Q, R, maxSteps=3000, leader_inps=True):

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

        # Noise trajectory
        self.Q = Q
        self.R = R

        # Controller to sample for the leader inputs, assuming it uses a PID
        self.leaderController = PIDStub(relativeState=False)
        self.leaderUs = []

        # Our controller and trajectory tracker
        self.controller = controller
        self.trajectory_tracker = trajectory_tracker

        # Default args for bicycle kinematics for now
        w=1
        # Bounding box for rrt collision checking
        self.vpolygon = Polygon2([(-w/2, w/2), (-w/2, -w/2), (w/2, -w/2), (w/2, w/2)])
        self.vehicle = Unicycle(W=w, steer_max=1.413716694115407, polygon=self.vpolygon)

    # Set up the problem by calculating the observations for the first one hundred timesteps
    # during which the ego vehicle simply waits
    # Pushes everything into the trajectory tracker and returns nothing
    def init(self):
        init_pos = self.start_pos# + np.random.multivariate_normal(np.zeros(3), self.Q)
        # List for tracking inputs, observations
        egoView = []
        leaderV =[]
        us = []
        leaderUs = []
        leaderNoise = np.array([[0.0, 0.0],
                                [0.0,0.0]])
        # We are stationary for the first 100 timesteps
        u = np.array([0.0,0.0])
        for i in range(self.offset):
            # Calculate obervation with noise
            egoView.append(global2LocalCoords(init_pos,self.traj.eval((i)*self.dt)))
            #egoView[i] += np.random.multivariate_normal(np.zeros(3), np.zeros(()self.Q)
            # Calculate leader inputs
            leaderU = self.leaderController.demand(self.traj.eval((i- 1)*self.dt),[self.traj.eval((i)*self.dt)], leaderNoise)(None, None, None)
            leaderUs.append(leaderU)
            us.append(u)
        # Push initial observations into the trajectory tracker
        self.trajectory_tracker.push_init(self.start_pos, egoView, us, leaderUs)

    # Run the simulations and return the givven inputs
    def simulate(self, debug=False):
        states = []
        initInput = np.array([0.0,0.0])
        inputs = [initInput]
        traj   = []
        leaderNoise = np.array([[0.0, 0.0],
                                [0.0,0.0]])
        s = self.vehicle._x
        # Loop through for the maximum amount of time or until the goal is reached
        if debug:
            self.max_length = 250
        #self.max_length=100
        for i in range(self.max_length):
            # Calculate the observation with noise
            obs = global2LocalCoords(s,self.traj.eval((i+self.offset)*self.dt), shiftToPiRange=False) #+ np.random.multivariate_normal(np.zeros(3), self.Q)

            # Get the true trajectory(used for tracking and stubs)
            true_traj = [self.traj.eval((j)*self.dt) for j in range(i, i+100)]

            # Get the leader controller inputs for leader t = ego_t + 100*dt
            leaderU = self.leaderController.demand(self.traj.eval((i+self.offset - 1)*self.dt),[self.traj.eval((i+self.offset)*self.dt)], leaderNoise)(None, None, None)

            # Get our estimated trajectory from the tracker
            # We note that this is cast to a vector in the wrapper to simplify type handling in the controllers
            est, covs = self.trajectory_tracker.step(obs, s, inputs[-1], leaderU, true_traj)

            # Sample inputs
            u   = self.controller.demand(s, est, noise=np.zeros((2,2)))#self.R)
            # Step the vehicle model according to our input
            odo = self.vehicle.step(u)

            s = deepcopy(self.vehicle._x)
            s[2] = wrap(s[2])
            # Track the states, etc.
            traj.append(self.traj.eval(i*self.dt))
            states.append(s)
            inputs.append(u(None, None, None))
            # Debug plotting
            if debug and i > 225:
                plt.clf()
                # Plot the trajectories
                ego_traj = np.array([global2LocalCoords(s, true_pos) for true_pos in true_traj])
                plt.plot(ego_traj[:,0], ego_traj[:,1], label='True Trajectory')
                plt.plot(est[:,0], est[:,1], label='Pred Trajectory')
                plt.plot(est[0,0], est[0,1], 'b*')
                plt.plot(ego_traj[0,0], ego_traj[0,1], 'r*')
                plt.title('Trajectory')
                plt.legend()
                plt.show()
                # Plot thetas
                plt.title('Thetas')
                plt.plot(n.array(states)[:,2], label='True Theta')
                if len(est.shape) == 4:
                    plt.plot(np.arctan2(est[:,2], est[:,3]), label='Pred Theta')
                else:
                    plt.plot(est[:,2], label='Pred Theta')

                plt.plot(np.arctan2(pred_st[:,1,0], pred_st[:,0,0]))
                plt.legend()
                plt.show()

            # Break if we get close to the goal position
            if np.linalg.norm(np.array(s) - self.goal_pos) <= 5.0 or np.linalg.norm(np.array(traj[-1])- self.goal_pos) <= 5.0:
                break

        return np.array(states), np.array(inputs), np.array(traj)

    def reset(self, seed=0):
        plt.cla()
        signal.signal(signal.SIGALRM, RRTTimeoutHandler) 
        signal.alarm(420)
        np.random.seed(seed)
        random.seed(seed)
        try:
            # Set random positions within the bounding space
            positions = (np.random.rand(2, 2) - 0.5) * (self.bound_x -5)*2
            thetas = np.random.rand(2,1) * 2.0 * np.pi 
            # Make sure the positions are far apart
            while np.linalg.norm(positions[0] - positions[1]) < 5.0 :
                positions = (np.random.rand(2, 2) - 0.5) * self.bound_x * 2.0
         
            self.start_pos = np.concatenate([positions[0], thetas[0]])
            self.goal_pos = np.concatenate([positions[1], thetas[1]])
            self.trajectory_tracker.reset()
         
         
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
            rrt = RRTPlanner(map=self.map, vehicle=self.vehicle, npoints=400, seed=seed)
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
        print(self.start_pos)

        # return the trajectory global view and the initial state
        return self.traj, self.start_pos

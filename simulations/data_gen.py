import numpy as np
from roboticstoolbox.mobile import Unicycle,PolygonMap, RRTPlanner,VehiclePolygon
from controllers.datagen.controller import *
from controllers.datagen.gmpc import *
from controllers.datagen.random import *
from spatialmath import Polygon2, Ellipse
from roboticstoolbox.tools.trajectory import *
import matplotlib.pyplot as plt
from klampt.model.trajectory import path_to_trajectory
from klampt.model import trajectory
from common.transformations import global2LocalCoords
import pandas as pd
import signal
import matplotlib.pyplot as plt
from copy import deepcopy
from common.util import RRTTimeoutHandler
from controllers.pid_stub import *

# This class contains an environment to generate a dataset for trajectory estimation
# in a convoy without without communication
class convoyDatasetGenerator():

    # Create a 400 x 400 unit space for reach avoid games
    def __init__(self, ):

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

        # Default args for bicycle kinematics for now
        self.w=1
        w = 1
        # Bounding box for rrt collision checking
        self.vpolygon = Polygon2([(-w/2, w/2), (-w/2, -w/2), (w/2, -w/2), (w/2, w/2)])
        self.vehicle = Unicycle(W=self.w, steer_max=1.413716694115407, polygon=self.vpolygon)
        # Generate a vehicle controller for the unicycle
        self.leaderController = PIDStub(relativeState=False)

    # Generate and return a single runs worth of data points
    def generateRun(self, view=True):
        # Run the simulation for up to 2000 seconds
        ret = self.vehicle.run(T=500, x0=self.start_pos)
        # Get the noisy state and inputs from the controllers
        noisy_s, noisy_u = self.vehicle.control.get_run_stats()
        self.vehicle.control.reset()
        egoView = []
        trajView = []
        leaderV =[]

        leaderNoise = np.array([[0.0, 0.0],
                                [0.0,0.0]])
        count = range(len(ret))
        for i in range(self.offset):
            egoView.append(global2LocalCoords(ret[0],self.traj.eval((i)*self.dt)))
            trajView.append(self.traj.eval((i)*self.dt))
            leaderV.append(self.leaderController.demand(self.traj.eval((i- 1)*self.dt),[self.traj.eval((i)*self.dt)], leaderNoise)(None, None, None))
        # Loop through and get the trajectory of the leader from 100(10 seconds) timesteps in the future
        for i in range(len(ret)):
            egoView.append(global2LocalCoords(ret[i],self.traj.eval((i+self.offset)*self.dt)))
            trajView.append(self.traj.eval((i+self.offset)*self.dt))
            leaderU = self.leaderController.demand(self.traj.eval((i+self.offset - 1)*self.dt),[self.traj.eval((i+self.offset)*self.dt)], leaderNoise)(None, None, None)
            leaderV.append(leaderU)
            # We break when we get close to the goal to avoid repeating data
            if np.linalg.norm(np.array(trajView[-1]) - self.goal_pos) <= 5.0:
                break

        # Slice the states and inputs to match up with the run length without repeats
        ret = np.array(ret)[:i+1]
        pre_ret = np.tile(ret[0,:], (100,1))
        ret = np.concatenate([pre_ret, ret], axis=0)
        noisy_s, noisy_u = np.array(noisy_s)[:i+1], np.array(noisy_u)[:i+1]
        pre_s = np.tile(noisy_s[0,:], (100,1))
        noisy_s = np.concatenate([pre_s, noisy_s], axis=0)
        leaderV = np.array(leaderV)[:i+101]
        count = range(len(ret))
        # Plot the various trajectories for debugging and visualization purposes
        if view:
            plt.cla()
            plt.plot(ret[:,0], ret[:,1], 'r-')
            plt.plot(np.array(self.traj.milestones)[:,0], np.array(self.traj.milestones)[:,1], 'c-')
            plt.plot(np.array(trajView)[:,0], np.array(trajView)[:,1], 'b--')
            plt.show()

        egoView = np.array(egoView)
        trajView = np.array(trajView)
        noisy_u = np.concatenate([np.zeros((100,2)), noisy_u], axis=0)
        # Construct a dataframe object from our data points
        d = {'Step' :np.array(count), 'GlobalTrueX':ret[:,0], 'GlobalTrueY':ret[:,1], 'GlobalTrueTh':ret[:,2], 
             'GlobalX':noisy_s[:,0], 'GlobalY':noisy_s[:,1], 'GlobalTh':noisy_s[:,2],
             'v1':noisy_u[:,0],'v2': noisy_u[:,1],
             'GlobalTrajX': trajView[:,0], 'GlobalTrajY':trajView[:,1], 'GlobalTrajTh':trajView[:,2], 
             'EgoTrajX':egoView[:,0], 'EgoTrajY':egoView[:,1], 'EgoTrajTh':egoView[:,2], 'Leaderv1': leaderV[:,0] , 'Leaderv2':leaderV[:,1]}
        df = pd.DataFrame(data=d)

        return df, noisy_s 

    def reset(self, view=False, control=RandomController):

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
         
            self.vehicle = Unicycle(W=self.w, steer_max=1.413716694115407, polygon=self.vpolygon)
            # Generate a vehicle controller for the unicycle
            self.vehicle.control = control(100, traj, vehicle=self.vehicle)

        except Exception as exc:
            print(exc)
            self.reset()
        signal.alarm(0)

        if view:
            rrt.plot(path)
            plt.show()

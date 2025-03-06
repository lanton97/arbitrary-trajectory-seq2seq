import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from ..utils.enum_class import TrajType
import numpy as np
from roboticstoolbox.mobile import Unicycle,PolygonMap, RRTPlanner,VehiclePolygon
from roboticstoolbox.tools.trajectory import *
from klampt.model.trajectory import path_to_trajectory
from klampt.model import trajectory
from common.transformations import global2LocalCoords
from copy import deepcopy
from controllers.pid_stub import *
from spatialmath import Polygon2, Ellipse
import signal
from common.util import RRTTimeoutHandler

def generateArbitraryTraj(mean_obst_rad=7,obst_std=1.414, num_obstacles=5, boundary=50, dt=0.1, seed=0, v_max=0.2):

    plt.cla()
    signal.signal(signal.SIGALRM, RRTTimeoutHandler) 
    signal.alarm(420)
    np.random.seed(seed)
    try:
        # Set random positions within the bounding space
        positions = (np.random.rand(2, 2) - 0.5) * (boundary -5)*2
        thetas = np.random.rand(2,1) * 2.0 * np.pi 
 
        w=1
        # Bounding box for rrt collision checking
        vpolygon = Polygon2([(-w/2, w/2), (-w/2, -w/2), (w/2, -w/2), (w/2, w/2)])
        vehicle = Unicycle(W=w, steer_max=1.413716694115407, polygon=vpolygon)
 
        # Make sure the positions are far apart
        while np.linalg.norm(positions[0] - positions[1]) < 5.0 :
            positions = (np.random.rand(2, 2) - 0.5) * boundary * 2.0
 
        start_pos = np.array([0.0,0.0,0.0]) 
        goal_pos = np.concatenate([positions[1], thetas[1]])
 
 
        # Generate a map of obstacles for motion planning
        env = PolygonMap(workspace=[-boundary,boundary])
 
        obst_rads = np.random.normal(loc=mean_obst_rad, scale=obst_std, size=(num_obstacles))
        obst_positions = (np.random.rand(2, num_obstacles) - 0.5) * boundary * 2.0
        for i in range(num_obstacles):
            Polygon = []
            # Create a circle from 32 points
            for j in range(4):
                theta = 2*np.pi * j / 32
                Polygon.append((obst_positions[0,i] + obst_rads[i]*np.cos(theta), obst_positions[1,i] + obst_rads[i]*np.sin(theta)))
            env.add(Polygon)
 
        # RRT planning for a path
        rrt = RRTPlanner(map=env, vehicle=vehicle, npoints=400, seed=seed)
        rrt.plan(goal=goal_pos, showsamples=True, showvalid=False)
        path, status = rrt.query(start=start_pos)
        # Create a trajectory from the path
        traj = trajectory.Trajectory(milestones=path.tolist())
        #traj = path_to_trajectory(traj, vmax=0.8*v_max, velocities='auto', dt=dt/2)
        traj = path_to_trajectory(traj, vmax=0.5, velocities='auto', dt=dt/2)
 
        plt.clf()
    except Exception as exc:
        print(exc)
        traj = generateArbitraryTraj(mean_obst_rad,obst_std, num_obstacles, boundary, dt, seed)
    signal.alarm(0)

    # return the trajectory global view and the initial state
    return traj


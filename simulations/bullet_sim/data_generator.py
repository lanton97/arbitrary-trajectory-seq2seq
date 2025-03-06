import numpy as np
from .utils.enum_class import TrajType, ControllerType, EnvType
from controllers.gmpc import GeometricMPC
from .planner.ref_traj_generator import *
from .gen_bullet_data import generateBulletData
import os
import matplotlib.pyplot as plt
from .environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from .environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini
from controllers.nonlinear_mpc import *

def generateBulletRun(gui=False, dt=0.02, maxSteps=5000, view=True, seed=0):
    # Set up matrices for sampling noise
    Q = np.array([[0.02, 0.0, 0.0],
              [0.0,  0.02,0.0],
              [0.0,  0.0,0.02]])
    R = np.array([[0.02, 0.0],
              [0.0,  0.02]])

    # Set relative state on or off
    rel_state = True

    # set env and traj
    env = ScoutMini(gui=gui, debug=True, dt=dt)
    v_min, v_max, w_min, w_max = env.get_vel_cmd_limit()

    controller = GeometricMPC(relativeState=rel_state, dt=dt)
    # set controller limits
    controller.set_control_bound(v_min, v_max, w_min, w_max)

    traj = generateArbitraryTraj(dt=dt, v_max=v_max, seed=seed)
    init_state = traj.eval(0)
    init_state = np.array([init_state[0], init_state[1], 0.])

    env = ScoutMini(gui=gui, debug=True, init_state=init_state, dt=dt)

    robotVels = env.robot_vels
    df, _ = generateBulletData(init_state, controller, traj, 
                                                   env, Q, R, maxSteps=maxSteps, gui=gui, dt=dt, view=view)

    return df


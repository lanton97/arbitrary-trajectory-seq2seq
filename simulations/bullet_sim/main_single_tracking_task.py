import numpy as np
from .utils.enum_class import TrajType, ControllerType, EnvType
from controllers.gmpc import GeometricMPC
from .planner.ref_traj_generator import *
from .monte_carlo_test_turtlebot import simulation
import os
import matplotlib.pyplot as plt
from .environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from .environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini

def bullet_tracking(trajectory_tracker, Q,R, env_type, gui=False, dt=0.02, maxSteps=3000, seed=0, debug=False):

    # Set relative state on or off
    rel_state = True

    # set env and traj
    if env_type == EnvType.TURTLEBOT:
        env = Turtlebot(gui=False, debug=True, dt=dt)
    elif env_type == EnvType.SCOUT_MINI:
        env = ScoutMini(gui=False, debug=True, dt=dt)
    else:
        raise NotImplementedError
    v_min, v_max, w_min, w_max = env.get_vel_cmd_limit()

    controller = GeometricMPC(relativeState=rel_state, dt=dt)
    # set controller limits
    controller.set_control_bound(v_min, v_max, w_min, w_max)

    traj = generateArbitraryTraj(dt=dt,seed=seed, v_max=v_max)
    init_state = traj.eval(0)
    init_state = np.array([init_state[0], init_state[1], 0.])

    if env_type == EnvType.TURTLEBOT:
        env = Turtlebot(gui=gui, debug=True, init_state=init_state, dt=dt)
    elif env_type == EnvType.SCOUT_MINI:
        env = ScoutMini(gui=gui, debug=True, init_state=init_state, dt=dt)
    else:
        raise NotImplementedError


    store_SE2, store_twist, trajectory, _, _= simulation(init_state, controller, traj, trajectory_tracker,
                                                   env, Q, R, maxSteps=maxSteps, gui=gui, debug=debug, dt=dt)

    return np.array(store_SE2), np.array(store_twist), np.array(trajectory)

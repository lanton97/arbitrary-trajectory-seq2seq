from .environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from .environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini
import numpy as np
from .utils.enum_class import TrajType, ControllerType, EnvType
from controllers.gmpc import GeometricMPC
from controllers.pid_stub import *
from manifpy import SE2, SO2
import matplotlib.pyplot as plt
import scipy
from common.transformations import *
from copy import deepcopy
import pandas as pd

def generateBulletData(init_state, controller, trajectory, env, Q,R,
               gui=False, offset=100, maxSteps=5000, dt=0.02, view=True):

    leader_controller = PIDStub()
    inputs = []
    states = []
    noisy_s = []
    traj = []
    curr_state = env.get_state()
    egoView = []

    # First do the first x seconds to push initial data to the trajectory tracker

    # List for tracking inputs, observations
    us = []
    leaderUs = []
    leaderNoise = np.array([[0.0, 0.0],
                            [0.0,0.0]])
    # We are stationary for the first 100 timesteps
    u = np.array([0.0,0.0])
    for i in range(offset):
        # Calculate obervation with noise
        egoView.append(global2LocalCoords(curr_state,trajectory.eval((i)*dt)))
        egoView[i] += np.random.multivariate_normal(np.zeros(3), Q)
        traj.append(trajectory.eval((i)*dt))
        # Calculate leader inputs
        leaderU = leader_controller.demand(trajectory.eval((i- 1)*dt),[trajectory.eval((i)*dt)], leaderNoise)(None, None, None)
        leaderUs.append(leaderU)
        us.append(u)

    # Now we Loop through to max steps to evaluate the controller-tracker pair
    for i in range(maxSteps):
        traj.append(trajectory.eval((i+offset)*dt))
        curr_state = env.get_state()
        states.append(curr_state)
        inputs.append(env.get_twist())
        noisy_state = curr_state +  np.random.multivariate_normal(np.zeros(3), Q)

        noisy_s.append(noisy_state)

        # Our sim code

        obs = global2LocalCoords(curr_state,trajectory.eval((i+offset)*dt), shiftToPiRange=False) 
        obs += np.random.multivariate_normal(np.zeros(3), Q)
        egoView.append(obs)

        # Get the true trajectory(used for tracking and stubs)
        true_traj = [trajectory.eval((j)*dt) for j in range(i, i+100)]

        ego_traj = np.array([global2LocalCoords(curr_state, true_pos) for true_pos in true_traj])
        # Get the leader controller inputs for leader t = ego_t + 100*dt
        leaderU = leader_controller.demand(trajectory.eval((i+offset - 1)*dt),[trajectory.eval((i+offset)*dt)], leaderNoise)(None, None, None)
        leaderUs.append(leaderU)

        # Sample inputs
        vel_cmd  = controller.demand(curr_state, ego_traj, noise=R)(None, None, None)

        # Back to base simulation code
        twist = np.array([vel_cmd[0], 0, vel_cmd[1]])
        env.step(env.twist_to_control(twist))


    # Slice the states and inputs to match up with the run length without repeats
    states = np.array(states)[:i+1]
    pre_ret = np.tile(states[0,:], (100,1))
    states = np.concatenate([pre_ret, states], axis=0)
    noisy_s = np.array(noisy_s)[:i+1]
    pre_s = np.tile(noisy_s[0,:], (100,1))
    noisy_s = np.concatenate([pre_s, noisy_s], axis=0)
    leaderUs = np.array(leaderUs)[:i+101]
    count = range(len(states))
    inputs = np.array(inputs)
    inputs = np.concatenate([np.zeros((100,2)), inputs], axis=0)
    # Plot the various trajectories for debugging and visualization purposes
    if view:
         plt.cla()
         plt.plot(states[:,0], states[:,1], 'r-')
         plt.plot(np.array(trajectory.milestones)[:,0], np.array(trajectory.milestones)[:,1], 'c-')
         plt.plot(np.array(traj)[:,0], np.array(traj)[:,1], 'b--')
         plt.show()

    egoView = np.array(egoView)
    traj = np.array(traj)

    print(len(states), len(noisy_s), len(inputs), len(traj), len(egoView), len(leaderUs))

    # Construct a dataframe object from our data points
    d = {'Step' :np.array(count), 'GlobalTrueX':states[:,0], 'GlobalTrueY':states[:,1], 'GlobalTrueTh':states[:,2], 
         'GlobalX':noisy_s[:,0], 'GlobalY':noisy_s[:,1], 'GlobalTh':noisy_s[:,2],
         'v1':inputs[:,0],'v2': inputs[:,1],
         'GlobalTrajX': traj[:,0], 'GlobalTrajY':traj[:,1], 'GlobalTrajTh':traj[:,2], 
         'EgoTrajX':egoView[:,0], 'EgoTrajY':egoView[:,1], 'EgoTrajTh':egoView[:,2], 'Leaderv1': leaderUs[:,0] , 'Leaderv2':leaderUs[:,1]}
    df = pd.DataFrame(data=d)

    return df, noisy_s 


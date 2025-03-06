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

def simulation(init_state, controller, trajectory, trajectory_tracker, env, Q,R,
               gui=False, offset=100, maxSteps=3000, dt=0.02, debug=True):


    leader_controller = PIDStub()

    # Calculate the ratio we should multiply the inputs by when predicting the trajectory
    # The seq2seq model is trained on 0.1 dt, so we need to downscale the values


    inputs = []
    states = []
    traj = np.array([trajectory.eval(i*dt) for i in range(maxSteps+offset)])

    obs_list = []
    true_obs_list = []

    env.draw_ref_traj(traj)
    traj = []

    curr_state = env.get_state()

    # First do the first x seconds to push initial data to the trajectory tracker

    # List for tracking inputs, observations
    egoView = []
    leaderV =[]
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
        # Calculate leader inputs
        leaderU = leader_controller.demand(trajectory.eval((i- 1)*dt),[trajectory.eval((i)*dt)], leaderNoise)(None, None, None)
        leaderUs.append(leaderU)
        us.append(u)
    # Push initial observations into the trajectory tracker
    trajectory_tracker.push_init(init_state, egoView, us, leaderUs)
    if debug:
        maxSteps=150

    # Now we Loop through to max steps to evaluate the controller-tracker pair
    for i in range(maxSteps):
        traj.append(trajectory.eval(i*dt))
        curr_state = env.get_state()
        states.append(curr_state)
        inputs.append(env.get_twist())

        # Our sim code

        obs = global2LocalCoords(curr_state,trajectory.eval((i+offset)*dt), shiftToPiRange=False) 
        true_obs_list.append(deepcopy(obs))
        obs += np.random.multivariate_normal(np.zeros(3), Q)
        obs_list.append(obs)

        # Get the true trajectory(used for tracking and stubs)
        true_traj = [trajectory.eval((j)*dt) for j in range(i, i+100)]

        ego_traj = np.array([global2LocalCoords(curr_state, true_pos) for true_pos in true_traj])
        # Get the leader controller inputs for leader t = ego_t + 100*dt
        leaderU = leader_controller.demand(trajectory.eval((i+offset - 1)*dt),[trajectory.eval((i+offset)*dt)], leaderNoise)(None, None, None)

        # Get our estimated trajectory from the tracker
        est, covs = trajectory_tracker.step(obs, curr_state, inputs[-1], leaderU, true_traj)
        if debug and i > 90:
            plt.clf()
            # Plot the trajectories
            ego_traj = np.array([global2LocalCoords(curr_state, true_pos) for true_pos in true_traj])
            plt.plot(ego_traj[:,0], ego_traj[:,1], label='True Trajectory')
            plt.plot(est[:,0], est[:,1], label='Pred Trajectory')
            plt.plot(est[0,0], est[0,1], 'b*')
            plt.plot(ego_traj[0,0], ego_traj[0,1], 'r*')

            plt.title('Trajectory')
            plt.legend()
            plt.show()

        # Sample inputs
        vel_cmd  = controller.demand(curr_state, est, noise=R)(None, None, None)

        # Back to base simulation code
        twist = np.array([vel_cmd[0], 0, vel_cmd[1]])
        env.step(env.twist_to_control(twist))


    return states, inputs, traj, obs_list, true_obs_list

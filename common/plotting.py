import numpy as np
import matplotlib.pyplot as plt
from common.se2 import *
from scipy.stats import chi2

def plot_rewards(loss, val_loss, save_dir='', save_suffix='', show=True, y_ax_txt='Negative Log Likelihood'):
    #avg = np.convolve(np.squeeze(losses), np.ones(5), 'valid') / 5

    fig = plt.figure()
    plt.plot(np.squeeze(loss), label="Loss")
    plt.plot(np.squeeze(val_loss), label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel(y_ax_txt)
    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'rew_plot' + save_suffix)#, format='eps')
    plt.clf()
    plt.close()

def plot_direct_trajectory(trajectory, true_traj, show=True, save_dir='', save_suffix=''):
    mans = convertTrajToManifolds(trajectory)
    vec_traj = []
    for man in mans:
        vec_traj.append(man.vector.listView)

    vec_traj = np.array(vec_traj)
    fig = plt.figure()
    plt.plot(vec_traj[:,0], vec_traj[:,1], 'b', label='Predicted Trajectory')
    plt.plot(vec_traj[0,0], vec_traj[0,1], 'b*')
    plt.plot(true_traj[:,0], true_traj[:,1], 'g', label='True Trajectory')
    plt.plot(true_traj[0,0], true_traj[0,1], 'g*')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'trajectory_plot' + save_suffix)#, format='eps')
    plt.clf()
    plt.close()

def plot_trajectory(trajectory, covs, true_traj, show=True, save_dir='', save_suffix=''):
    mans = convertTrajToManifolds(trajectory)
    vec_traj = []
    for man in mans:
        vec_traj.append(man.vector.listView)

    vec_traj = np.array(vec_traj)
    fig = plt.figure()
    plt.plot(vec_traj[:,0], vec_traj[:,1], 'b', label='Predicted Trajectory')
    plt.plot(vec_traj[0,0], vec_traj[0,1], 'b*')
    plt.plot(true_traj[:,0], true_traj[:,1], 'g', label='True Trajectory')
    plt.plot(true_traj[0,0], true_traj[0,1], 'g*')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'trajectory_plot' + save_suffix)#, format='eps')
    plt.clf()
    plt.close()

def plot_state_viz(trajectory, covs, vert_x_tick=[], show=True, save_dir='', save_suffix=''):
    mans = convertTrajToManifolds(trajectory)
    vec_traj = []
    for man in mans:
        vec_traj.append(man.vector.listView)

    vec_traj = np.array(vec_traj)
    fig = plt.figure()
    fig.suptitle('Prediction of Trajectory Components')
    t = 0.1*np.array(range(vec_traj.shape[0]))

    alpha = 0.1
    s1 = chi2.isf(alpha, 1)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t, vec_traj[:,0])

    sigma = np.sqrt(s1*covs[:, 0, 0].detach().numpy())
    plt.fill_between(t, -sigma[:] + vec_traj[:,0], sigma[:] + vec_traj[:,0], color='green', alpha=0.2)
    plt.ylabel(r'$x$ [m]')

    ax1 = plt.subplot(3, 1, 2)
    plt.plot(t, vec_traj[:,1])

    sigma = np.sqrt(s1*covs[:, 1, 1].detach().numpy())
    plt.fill_between(t, -sigma[:]+ vec_traj[:,1] , sigma[:] + vec_traj[:,1] , color='green', alpha=0.2)
    plt.ylabel(r'$y$ [m]')

    ax1 = plt.subplot(3, 1, 3)
    plt.plot(t, vec_traj[:,2])

    sigma = np.sqrt(s1*covs[:, 2, 2].detach().numpy())
    plt.fill_between(t, -sigma[:]+vec_traj[:,2] , sigma[:]+vec_traj[:,2] , color='green', alpha=0.2)
    plt.ylabel(r'$\theta$ [rads]')
    if len(vert_x_tick) > 0:
        for tick in vert_x_tick:
            plt.axvline(x=tick, color = 'r')

    plt.xlabel(r'$t$ [s]')

    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'var_plot' + save_suffix)#, format='eps')

    plt.clf()
    plt.close()



def plot_traj_error(trajectory, true_traj, show=True, save_dir='', save_suffix=''):
    mans = convertTrajToManifolds(trajectory)
    vec_traj = []
    for man in mans:
        vec_traj.append(man.vector.listView)

    vec_traj = np.array(vec_traj)
    true_traj = np.array(true_traj)
    fig = plt.figure()
    fig.suptitle('Errors in Prediction of Trajectory Components')
    t = 0.1*np.array(range(vec_traj.shape[0]))

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t, vec_traj[:,0] - true_traj[:,0])

    plt.ylabel(r'$x$ [m]')

    ax1 = plt.subplot(3, 1, 2)
    plt.plot(t, vec_traj[:,1] - true_traj[:,1])

    plt.ylabel(r'$y$ [m]')

    ax1 = plt.subplot(3, 1, 3)
    err = []
    true_man = convertTrajToManifolds(true_traj)
    for i in range(vec_traj.shape[0]):
        box_min = BoxMinus(mans[i], true_man[i])
        err.append(box_min[2])
    err = np.array(err)

    plt.plot(t, err)

    plt.ylabel(r'$\theta$ [rads]')
    plt.xlabel(r'$t$ [s]')

    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'error_plot' + save_suffix)#, format='eps')

    plt.clf()
    plt.close()

def plot_state_error_viz(trajectory, covs, true_traj, vert_x_tick=[], show=True, save_dir='', save_suffix=''):
    mans = convertTrajToManifolds(trajectory)
    vec_traj = []
    for man in mans:
        vec_traj.append(man.vector.listView)

    vec_traj = np.array(vec_traj)
    true_traj = np.array(true_traj)
    fig = plt.figure()
    fig.suptitle('Errors in Prediction of Trajectory Components')
    t = 0.1*np.array(range(vec_traj.shape[0]))

    alpha = 0.1
    s1 = chi2.isf(alpha, 1)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t, vec_traj[:,0] - true_traj[:,0])

    sigma = np.sqrt(s1*covs[:, 0, 0].detach().numpy())
    plt.fill_between(t, -sigma[:] + (vec_traj[:,0] - true_traj[:,0]), sigma[:] + (vec_traj[:,0] - true_traj[:,0]), color='green', alpha=0.2)
    plt.ylabel(r'$x$ [m]')

    ax1 = plt.subplot(3, 1, 2)
    plt.plot(t, vec_traj[:,1] - true_traj[:,1])

    sigma = np.sqrt(s1*covs[:, 1, 1].detach().numpy())
    plt.fill_between(t, -sigma[:]+ (vec_traj[:,1] - true_traj[:,1]), sigma[:] + (vec_traj[:,1]  - true_traj[:,1]), color='green', alpha=0.2)
    plt.ylabel(r'$y$ [m]')

    ax1 = plt.subplot(3, 1, 3)
    err = []
    true_man = convertTrajToManifolds(true_traj)
    for i in range(vec_traj.shape[0]):
        box_min = BoxMinus(mans[i], true_man[i])
        err.append(box_min[2])
    err = np.array(err)

    plt.plot(t, err)

    sigma = np.sqrt(s1*covs[:, 2, 2].detach().numpy())
    plt.fill_between(t, -sigma[:]+(err) , sigma[:]+ (err), color='green', alpha=0.2)
    plt.ylabel(r'$\theta$ [rads]')
    if len(vert_x_tick) > 0:
        for tick in vert_x_tick:
            plt.axvline(x=tick, color = 'r')

    plt.xlabel(r'$t$ [s]')

    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'error_plot' + save_suffix)#, format='eps')

    plt.clf()
    plt.close()

def plot_true_and_pred(trajectory, covs, true_traj, vert_x_tick=[], show=True, save_dir='', save_suffix=''):
    mans = convertTrajToManifolds(trajectory)
    vec_traj = []
    for man in mans:
        vec_traj.append(man.vector.listView)

    vec_traj = np.array(vec_traj)
    fig = plt.figure()
    fig.suptitle('Prediction of Trajectory Components and True Trajectories')
    t = 0.1*np.array(range(vec_traj.shape[0]))

    alpha = 0.1
    s1 = chi2.isf(alpha, 1)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t, vec_traj[:,0], 'b-')
    plt.plot(t, true_traj[:,0], 'r--')

    sigma = np.sqrt(s1*covs[:, 0, 0].detach().numpy())
    plt.fill_between(t, -sigma[:] + vec_traj[:,0], sigma[:] + vec_traj[:,0], color='green', alpha=0.2)
    plt.ylabel(r'$x$ [m]')

    ax1 = plt.subplot(3, 1, 2)
    plt.plot(t, vec_traj[:,1], 'b-')
    plt.plot(t, true_traj[:,1], 'r--')

    sigma = np.sqrt(s1*covs[:, 1, 1].detach().numpy())
    plt.fill_between(t, -sigma[:]+ vec_traj[:,1] , sigma[:] + vec_traj[:,1] , color='green', alpha=0.2)
    plt.ylabel(r'$y$ [m]')

    ax1 = plt.subplot(3, 1, 3)
    plt.plot(t, vec_traj[:,2], 'b-')
    plt.plot(t, true_traj[:,2], 'r--')

    sigma = np.sqrt(s1*covs[:, 2, 2].detach().numpy())
    plt.fill_between(t, -sigma[:]+vec_traj[:,2] , sigma[:]+vec_traj[:,2] , color='green', alpha=0.2)
    plt.ylabel(r'$\theta$ [rads]')
    if len(vert_x_tick) > 0:
        for tick in vert_x_tick:
            plt.axvline(x=tick, color = 'r')

    plt.xlabel(r'$t$ [s]')

    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'true_and_pred_state_plot' + save_suffix)#, format='eps')

    plt.clf()
    plt.close()

def get_min_change_x_tick(traj, delta=0.001):
    dt = 0.1
    vert_x_tick = []
    for i in range(1,len(traj)):
        pointA = traj[i-1]
        thA = pointA[2]
        pointB = traj[i]
        thB = pointB[2]

        if len(vert_x_tick) % 2 == 0 and abs(thA - thB) < delta:
            vert_x_tick.append(dt*i)
        elif len(vert_x_tick) % 2 == 1 and abs(thA - thB) > delta:
            vert_x_tick.append(dt*i)

    return vert_x_tick

def plot_control_effort(traj, true_traj, inputs, tstep=500, show=True, save_dir='', save_suffix=''):
    fig = plt.figure()
    tstep = min(inputs.shape[0], tstep)

    t = 0.1*np.array(range(tstep))

    mans = convertTrajToManifolds(traj)
    vec_traj = []
    for man in mans:
        vec_traj.append(man.vector.listView)

    vec_traj = np.array(vec_traj)
    true_traj = np.array(true_traj)

    ax1 = plt.subplot(5, 1, 1)
    plt.plot(t, vec_traj[:tstep,0], 'b-')
    plt.plot(t, true_traj[:tstep,0], 'r--')

    plt.ylabel(r'$x$ [m]')

    ax1 = plt.subplot(5, 1, 2)
    plt.plot(t, vec_traj[:tstep,1], 'b-')
    plt.plot(t, true_traj[:tstep,1], 'r--')

    plt.ylabel(r'$y$ [m]')

    ax1 = plt.subplot(5, 1, 3)
    err = []

    plt.plot(t, vec_traj[:tstep,2], 'b-')
    plt.plot(t, true_traj[:tstep,2], 'r--')
    plt.ylabel(r'$\theta$ [rads]')

    plt.xlabel(r'$t$ [s]')

    ax1 = plt.subplot(5, 1, 4)
    plt.plot(t, inputs[:tstep,0], 'b-')
    plt.ylabel(r'$u_{1}$ [m/s]')

    ax2 = plt.subplot(5, 1, 5)
    plt.plot(t, inputs[:tstep,1], 'b-')
    plt.ylabel(r'$u_{2}$ [$\theta$/s]')

    plt.xlabel(r'$t$ [s]')
    fig.tight_layout()

    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'control_effort' + save_suffix)#, format='eps')

    plt.clf()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from common.se2 import *
# Calculate the forward and lateral velocities by looking at the heading of the vehicle at each timestep
def calcFwdLatVels(states, velocities):
    fwdVels = []
    latVels = []
    for state, vel in zip(states, velocities):
        heading = state[2]
        # First two components should be x, y vels, third is z
        vx, vy = vel[0], vel[1]

        forward = vx * np.cos(heading) + vy * np.sin(heading)
        lateral = -vx * np.sin(heading) + vy * np.cos(heading)

        fwdVels.append(forward)
        latVels.append(lateral)

    return np.array(fwdVels), np.array(latVels)

def plotFwdLatVels(fwdVels, latVels, errorInds=None, plotRegion=None):
    fig = plt.figure()
    fig.suptitle('Forward and Lateral Velocities')
    t = 0.02*np.array(range(fwdVels.shape[0]))
    if plotRegion is not None:
        inds = slice(*plotRegion)
    else:
        inds = slice(0, fwdVels.shape[0])

    ax1 = plt.subplot(2, 1, 1)
    plt.plot(t[inds], fwdVels[inds],'b-')

    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')
    plt.ylabel(r'$Forward Vel.$ [m/s]')

    ax1 = plt.subplot(2, 1, 2)
    plt.plot(t[inds], latVels[inds],'r--')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.ylabel(r'$Lateral Vel.$ [m/s]')
    plt.xlabel(r'$t$ [s]')

    plt.show()

    plt.clf()
    plt.close()


# Detect a change in RMSEs greater than a threshold and return the index at which it occurs
def detectErrorRegion(states, traj, thresh=0.5):
    # Error at t=0 is irrelevant so we take E at t=1 and iterate out
    prev_err = np.linalg.norm(states[1] - traj[1])
    for i in range(2, len(states)):
       curr_err = np.linalg.norm(states[i] - traj[i])
       if curr_err - prev_err > thresh:
           print('Error Jump at index: ' + str(i))
           return i
       prev_err = curr_err
    return -1


def plotTrueObservedRelStates(trueRelStates, obsRelStates,errorInds=None, plotRegion=None):
    trueRelStates = np.array(trueRelStates)
    obsRelStates = np.array(obsRelStates)
    fig = plt.figure()
    fig.suptitle('True and Observed Relative States')
    t = 0.02*np.array(range(trueRelStates.shape[0]))

    if plotRegion is not None:
        inds = slice(*plotRegion)
    else:
        inds = slice(0, trueRelStates.shape[0])

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t[inds], trueRelStates[inds,0],'r--')
    plt.plot(t[inds], obsRelStates[inds,0], 'b-')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.ylabel(r'$x$ [m]')

    ax1 = plt.subplot(3, 1, 2)
    plt.plot(t[inds], trueRelStates[inds,1],'r--')
    plt.plot(t[inds], obsRelStates[inds,1], 'b-')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.ylabel(r'$y$ [m]')

    ax1 = plt.subplot(3, 1, 3)
    plt.plot(t[inds], trueRelStates[inds,2],'r--')
    plt.plot(t[inds], obsRelStates[inds,2], 'b-')
    plt.ylabel(r'$\theta$ [rads]')
    plt.xlabel(r'$t$ [s]')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.show()

    plt.clf()
    plt.close()

def plotTrueObservedError(trueRelStates, obsRelStates, errorInds=None, plotRegion=None):
    trueRelStates = np.array(trueRelStates)
    obsRelStates = np.array(obsRelStates)
    fig = plt.figure()
    fig.suptitle('True and Observed Relative States')
    t = 0.02*np.array(range(trueRelStates.shape[0]))

    if plotRegion is not None:
        inds = slice(*plotRegion)
    else:
        inds = slice(0, trueRelStates.shape[0])

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(t[inds], trueRelStates[inds,0] - obsRelStates[inds,0])
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.ylabel(r'$x$ [m]')

    ax1 = plt.subplot(3, 1, 2)
    plt.plot(t[inds], trueRelStates[inds,1] - obsRelStates[inds,1])
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.ylabel(r'$y$ [m]')

    ax1 = plt.subplot(3, 1, 3)
    err = []
    for i in range(trueRelStates.shape[0]):
        box_min = BoxMinusVector(trueRelStates[i], obsRelStates[i])
        err.append(box_min[2])
    err = np.array(err)

    plt.plot(t[inds], err[inds])

    plt.ylabel(r'$\theta$ [rads]')
    plt.xlabel(r'$t$ [s]')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.show()

    plt.clf()
    plt.close()

def plotGains(iekfWrapper,errorInds=None, plotRegion=None):
    iekf = iekfWrapper.iekf
    egoGains = np.array(iekf.egoTracker.Ks)
    leaderGains = np.array(iekf.leaderTracker.Ks)
    if plotRegion is not None:
        inds = slice(*plotRegion)
    else:
        inds = slice(0, egoGains.shape[0])

    fig = plt.figure()
    fig.suptitle('IEKF Gains')
    t=0.02*np.array(range(egoGains.shape[0]))
    print(egoGains.shape)

    plt.subplot(3,2,1)
    plt.plot(t[inds], egoGains[inds,0,0])
    plt.ylabel('x Kalman Gains')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,2)
    plt.plot(t[inds], leaderGains[inds,0,0])
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,3)
    plt.plot(t[inds], egoGains[inds,1,1])
    plt.ylabel('y Kalman Gains')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,4)
    plt.plot(t[inds], leaderGains[inds,1,1])
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,5)
    plt.plot(t[inds], egoGains[inds,2,1])
    plt.ylabel('Theta Kalman Gains')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,6)
    plt.plot(t[inds], leaderGains[inds,2,1])
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.show()


def plotCovs(iekfWrapper,errorInds=None, plotRegion=None):
    iekf = iekfWrapper.iekf
    egoCovs = np.array(iekf.egoTracker.sigmas)
    leaderCovs = np.array(iekf.leaderTracker.sigmas)
    if plotRegion is not None:
        inds = slice(*plotRegion)
    else:
        inds = slice(0, egoCovs.shape[0])

    fig = plt.figure()
    fig.suptitle('IEKF Covs')
    t=0.02*np.array(range(egoCovs.shape[0]))

    plt.subplot(3,2,1)
    plt.plot(t[inds], egoCovs[inds,0,0])
    plt.ylabel('x Kalman Covs')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,2)
    plt.plot(t[inds], leaderCovs[inds,0,0])
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,3)
    plt.plot(t[inds], egoCovs[inds,1,1])
    plt.ylabel('y Kalman Covs')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,4)
    plt.plot(t[inds], leaderCovs[inds,1,1])
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,5)
    plt.plot(t[inds], egoCovs[inds,2,2])
    plt.ylabel('Theta Kalman Covs')
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.subplot(3,2,6)
    plt.plot(t[inds], leaderCovs[inds,2,2])
    if errorInds is not None:
        plt.axvline(x=errorInds, color = 'r')

    plt.show()

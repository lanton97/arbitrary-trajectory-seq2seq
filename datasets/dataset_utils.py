import pandas as pd
from common.transformations import global2LocalCoords
import torch
import numpy as np

# Attributes we want to select from each DS
standard_attributes = ['Step', 'GlobalTrueX', 'GlobalTrueY','GlobalTrueTh' , 'GlobalX', 'GlobalY', 'GlobalTh', 'v1', 'v2', 'GlobalTrajX', 'GlobalTrajY', 'GlobalTrajTh','EgoTrajX', 'EgoTrajY', 'EgoTrajTh', 'Run', 'Leaderv1', 'Leaderv2']
# Attributes used for the inputs of the NN
# Theta is preprocessed and handled below
input_attributes = ['EgoTrajX', 'EgoTrajY']
# Inputs for the ego and leader vehicles
vehicle_inputs = ['v1','v2', 'Leaderv1', 'Leaderv2']
# Items needed for IEKF tracking
iekf_attributes  = ['GlobalTrueX', 'GlobalTrueY','GlobalX', 'GlobalY', 'GlobalTh', 'GlobalTrajX', 'GlobalTrajY', 'GlobalTrajTh', 'EgoTrajX', 'EgoTrajY', 'EgoTrajTh', 'v1', 'v2', 'Leaderv1', 'Leaderv2','GlobalTrueTh']
# Items for IEKF target constructions
iekf_target_attr  = ['GlobalX', 'GlobalY', 'GlobalTh']
iekf_target_vels  = ['v1', 'v2']
iekf_target_obs   = ['EgoTrajX', 'EgoTrajY', 'EgoTrajTh']

# Read a fixed dataaset
def load_dataset(dataset_str="datasets/convoy.csv"):
    df = pd.read_csv(dataset_str, index_col=False)
    return df

# Select a singl full trajectory
def select_run(df, run=0):
    run = df.loc[df['Run'] == run]
    return run

# Select mulitple trajectories in a range
def select_runs(df, runs=(0, 50)):
    runs = df.between(runs[0], runs[1])

# Select features from the dataset
def select_features(df, features=standard_attributes):
    df = df[standard_attributes]
    return df

# Here, we split the Dataset into input windows of size windowsize
# options included for overlapping, single runs for debugging/visualization
def splitIntoWindows(df, windowSize=200, step=100, overlap=False, singleRun=False): 
    if singleRun:
        num_runs = 1
    else:
        num_runs = getLastValue(df, 'Run')

    windows = []
    # Loop through each run
    for i in range(num_runs):
        # Choose the relevant run
        run = select_run(df, run=i)
        # Get the length of the run
        num_steps = getLastValue(run, 'Step')
        step = step if overlap else windowSize
        # Loop through at a fixed number of steps in overlaps
        for j in range(0, num_steps, step):
            # If we don;t have enough data, simple exit
            if j + windowSize > num_steps:
                break
            # Build the input/target window
            window = constructWindow(run, windowSize, j, 'Step')
            windows.append(window)

    return windows

# Select a single window
def constructWindow(df, size, startIndex, column):
    # Choose a fixed number of rows and features
    window = df[df[column].between(startIndex, startIndex+size - 1)]
    # Drop the superfluous information
    window = window.drop(['Run', 'Step'], axis=1)
    return window

# return the final value in a column
def getLastValue(df, column):
    tail = df.tail(n=1)[column]
    num_runs = tail.iloc[0]
    return num_runs

# Returns the features for the IEKF testing
def getIEKFFeatures(windows):
    features = []
    for window in windows:
        target = calculateTrajectoryViewFromPoint(window)
        if not isValid(target):
            continue
        feature = iekfSelect(window)
        features.append(feature)

    return features

# Returns the features for the IEKF testing
def getIEKFTargets(windows):
    features = []
    for window in windows:
        target = calculateTrajectoryViewFromPoint(window)
        if not isValid(target):
            continue
        feature = iekfTrgSelect(window)
        features.append(feature)

    return features

def getIEKFTargetFeatures(windows):
    features = []
    for window in windows:
        target = calculateTrajectoryViewFromPoint(window)
        if not isValid(target):
            continue
        feature = iekfSelect(window)
        features.append(feature)

    return features

# Create inputs and targets for training
def constructInputTargetPairs(windows, leader_speed=True):
    targets = []
    inputs = []
    # Loop through each window
    for window in windows:
        # Create the target(trajectory) from the final ego pos
        target = calculateTrajectoryViewFromPoint(window)
        # Filter out places where the vehicles don't move
        if not isValid(target):
            continue

        targets.append(target)
        # Create an input window for the NN
        inp = windowToInput(window, leader_speed=leader_speed)
        inputs.append(inp)

    return inputs, targets

# Check that the first and last position are a minimum distance apart
def isValid(inputs, distance_threshold=0.3):
    pos1 = inputs[0,0:2]
    pos2 = inputs[-1,0:2]
    dist = np.linalg.norm(pos2 - pos1)
    return dist>distance_threshold

# Generate the trajectory we want to output
def calculateTrajectoryViewFromPoint(window):
    num_points = len(window)
    view_point = window.iloc[num_points // 2-1]
    view_pose = view_point[['GlobalX', 'GlobalY', 'GlobalTh']]
    trajectoryLocalView = []
    # Loop through the window
    for i in range(num_points // 2):
        #Calculate the ego position from where we are
        globalTrajPoint = window[['GlobalTrajX', 'GlobalTrajY', 'GlobalTrajTh']].iloc[i]
        localView = global2LocalCoords(view_pose, globalTrajPoint, shiftToPiRange=False)
        trajectoryLocalView.append(localView)
    return torch.Tensor(np.array(trajectoryLocalView))

# Convert a window into a NN input
def windowToInput(window, leader_speed=True):
    # Calculate the size of our input window
    num_points = len(window)
    num_inps = num_points // 2
    # Select the first X items
    inp = window[input_attributes].head(num_inps + 1).to_numpy()
    # get the thetas
    th = np.expand_dims(window['EgoTrajTh'].head(num_inps + 1).to_numpy(), 1)
    # Get leader and ego inputs
    vs = window[vehicle_inputs].head(num_inps + 1).to_numpy()
    leader_inp = window[input_attributes].tail(num_inps + 1).to_numpy()
    leader_th = np.expand_dims(window['EgoTrajTh'].tail(num_inps + 1).to_numpy(), 1)
    # Construc the input tensor
    inp = np.concatenate([inp, np.cos(th), np.sin(th), vs], axis=1)

    return inp[:-1]

def iekfSelect(window):
    num_points = len(window)
    num_inps = num_points // 2
    features = window[iekf_attributes].head(num_inps + 1)
    features = features.to_numpy()
    return features[:-1]

iekf_target_attr  = ['GlobalX', 'GlobalY', 'GlobalTh']
iekf_target_vels  = ['v1', 'v2']
iekf_target_obs   = ['EgoTrajX', 'EgoTrajY', 'EgoTrajTh']

def iekfTrgSelect(window):
    num_points = len(window)
    num_inps = num_points // 2
    poses = window[iekf_target_attr].head(num_inps + 1).to_numpy()[:-1]
    inps = window[iekf_target_vels].head(num_inps + 1).to_numpy()[:-1]
    obs = window[iekf_target_obs].head(num_inps + 1).to_numpy()[:-1]
    return (poses, inps, obs)


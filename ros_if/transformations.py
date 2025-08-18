import math
import numpy as np
# Wrap theta to (-pi, pi)
def wrap(theta):
    if abs(theta) > np.pi:
        revolutions = math.floor(( theta + np.pi ) * ( 1.0 /(2*np.pi) ))
        theta -= revolutions * 2*np.pi
    return theta 


# Convert a global position or pose to local coordinates
# Base coords are a position or pose, global_coords is also a position or pose
# positions are two values, x, y, and poses are three, x, y, theta 
def global2LocalCoords(base_coords, global_coords, shiftToPiRange=False):
    # We set the rotation to zero by default
    theta = wrap(base_coords[2])
    # Calculate the new theta value of the viewed item in the ego view
    newTheta = -(theta - global_coords[2])
    if shiftToPiRange:
        newTheta -= np.pi

    # Wrap the new theta as well
    newTheta = wrap(newTheta)

    # Calculate the rotation and translation matrix to do the coordinate system transform
    rot_mat = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta),  np.cos(theta)],
            ])

    trn_mat = np.array([
            [1.0, 0.0, -base_coords[0]],
            [0.0, 1.0, -base_coords[1]],
            [0.0, 0.0, 1.0]
            ])

    # Perform translation and then rotation on the object in homogenous coordinate system
    tmp_global_coords = np.concatenate((global_coords[0:2], [1.0]))
    transformed_coords = (trn_mat @ tmp_global_coords)
    transformed_coords = rot_mat @ transformed_coords[:2].T

    # Append the previously calculated direction of the pose
    new_coords = np.concatenate([transformed_coords[0:2], [newTheta]])

    return new_coords

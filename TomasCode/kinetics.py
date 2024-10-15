# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:07:48 2024.

@author: tm4dd

This module contains functions that perform kinetic (and kinematic) 
procedures.

"""
import numpy as np


# %% Get angles (from transformation matrix)


def get_angles(T):
    """
    Extract XYZ Cardan angles given a transformation matrix.

    Parameters
    ----------
    T : matrix (4 x 4)
        Transformation matrix (4 x 4).

    Returns
    -------
    Ox : float
        Angle about x-axis (roll)
    Oy : float
        Angle about y-axis (pitch)
    Oz : float
        Angle about z-axis (yaw)

    """
    Ox = 1  # angle about x-axis
    Oy = 1  # angle about y-axis
    Oz = 1  # angle about z-axis
    return [Ox, Oy, Oz]  # return XYZ Cardan angles


# %% Get gait events


def get_gait_events(Fx, Fz, threshold):
    """
    Get gait events.

    Given anterior-posterior and vertical ground reaction forces (GRFs), obtain
    frames corresponding to initial contact, toe-off, midstance, and the start
    of propulsion (point at which anterior-posterior GRF becomes positive,
    i.e., in the anterior direction).

    Parameters
    ----------
    Fx : array
        Anterior-posterior GRF (N).
    Fz : array
        Vertical GRF (N).
    threshold : int
        Threshold of vertical GRF (N) that defines stance phase. The stance
        phase is defined as the time during which vertical GRF exceeds the
        threshold.

    Returns
    -------
    list
        ic = initial contact, to = toe-off, ms = midstance, prop_start = start
        of propulsion.

    """
    # Get initial contact
    # Find where vertical GRF first exceeds threshold
    # Start at max vertical GRF and search backward
    max_index = np.argmax(Fz)  # index of max vertical GRF
    i = max_index  # start at index of max vertical GRF
    while i > 0:
        i -= 1  # increment index backwards
        if Fz[i] <= threshold:  # search for first frame at or below threshold
            break  # stop search
    ic = i + 1  # initial contact = first frame above threshold

    # Get toe-off
    # Find where force after initial contact drops below threshold
    # Start at max force and search until the force drops below threshold
    to = 0  # initialize toe-off variable
    for i in range(max_index, len(Fz)):
        if (
            Fz[i] < threshold
        ):  # search for point at which force drops below threshold
            to = i - 1  # set toe-off to last index before the force drops
            # below threshold
            break

    # Get midstance: frame halfway between initial contact and toe-off
    ms = int(np.mean([ic, to]))

    # Get start of propulsion (alternate definition of midstance)
    # Find where anterior-posterior GRF crosses zero
    # Start at minimum and look for zero crossing
    prop_start = 0  # initialize start of propulsion variable
    min_index = np.argmin(Fx)  # index of minimum anterior-posterior GRF
    for i in range(min_index, len(Fx)):
        if Fx[i] >= 0:  # search for zero crossing
            prop_start = i  # set start of propulsion to current index
            break

    return [ic, to, ms, prop_start]


# %% Transformation matrices


def transform_matrix(Ax, Ay, Az, Bx, By, Bz, Ao, Bo):
    """
    Transform reference frame B to frame A.

    Multiply a vector expressed in frame B by the transformation matrix T to
    transform it into frame A (express it in frame A).

    Parameters
    ----------
    Ax : array
        Frame A x-axis expressed in global reference frame.
    Ay : array
        Frame A y-axis.
    Az : array
        Frame A z-axis.
    Bx : array
        Frame B x-axis expressed in global reference frame.
    By : array
        Frame B y-axis.
    Bz : array
        Frame B z-axis.
    Ao : array
        Position of frame A origin expressed in global reference frame.
    Bo : array
        Position of frame B origin expressed in global reference frame.

    Returns
    -------
    T : array
        Transformation matrix (4 x 4) to transform frame B to frame A.

    """
    # Transformation matrix: frame B to frame A
    T11 = np.dot(Ax, Bx)
    T21 = np.dot(Ax, By)
    T31 = np.dot(Ax, Bz)
    T12 = np.dot(Ay, Bx)
    T22 = np.dot(Ay, By)
    T32 = np.dot(Ay, Bz)
    T13 = np.dot(Az, Bx)
    T23 = np.dot(Az, By)
    T33 = np.dot(Az, Bz)
    p = Bo - Ao  # position vector from frame A origin to frame B origin
    T = np.array(
        [
            [T11, T12, T13, p[0]],
            [T21, T22, T23, p[1]],
            [T31, T32, T33, p[2]],
            [0, 0, 0, 1],
        ]
    )  # transformation matrix
    return T

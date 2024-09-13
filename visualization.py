# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:04:33 2024.

@author: tm4dd

Vizualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np

# %% Show markers in 3D plot


def show_markers(data, **options):
    """
    Display 3D marker positions for a given frame.

    Parameters
    ----------
    data : array
        marker position data (3 or 4 x N x k) where N = number of
        markers, k = number of frames
    (optional) frame : int
        Frame for which to plot 3D marker positions.
        Default = 0 for data with multiple frames.
        frame = 0 implies the first frame of the video data, etc.
    (optional) marker_index : int
        Index (in data array) of the marker to highlight.

    Returns
    -------
        None. Plots 3D marker positions.

    """
    frame = options.get("frame")
    if frame is None:
        frame = 0  # default to first frame if no argument given
    # x, y, z coordinates of markers
    if len(np.shape(data)) < 3:  # 2D data (e.g., mean static positions)
        x = data[0, :]
        y = data[1, :]
        z = data[2, :]
    else:
        x = data[0, :, frame]
        y = data[1, :, frame]
        z = data[2, :, frame]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(x, y, z, alpha=0.75, color="royalblue")

    # highlight the specified marker if a marker index is given
    marker_index = options.get("marker_index")
    if marker_index:
        # x, y, z position of marker
        if len(np.shape(data)) < 3:  # 2D data (e.g., mean static positions)
            x_marker = data[0, marker_index]
            y_marker = data[1, marker_index]
            z_marker = data[2, marker_index]
        else:
            x_marker = data[0, marker_index, frame]
            y_marker = data[1, marker_index, frame]
            z_marker = data[2, marker_index, frame]
        ax.scatter(x_marker, y_marker, z_marker, alpha=1, color="red")

    # Adjust scale so that axes are scale 1:1
    # z-axis: 0 to automatic z limit
    zlim = ax.get_zlim()[1]  # automatic z limit
    ax.set_zlim(0, zlim)
    # x-axis: +/- half z range from median position
    mid_x = np.median(x)
    half_z = zlim / 2  # half z range
    ax.set_xlim(mid_x - half_z, mid_x + half_z)
    # y-axis: +/- half z range from median position
    mid_y = np.median(y)
    ax.set_ylim(mid_y - half_z, mid_y + half_z)
    # Add axis labels
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Z position")

    # Verify scales are equal
    # print(ax.get_zlim()[1] - ax.get_zlim()[0])
    # print(ax.get_ylim()[1] - ax.get_ylim()[0])
    # print(ax.get_xlim()[1] - ax.get_xlim()[0])


# %% Show x, y, z positions of a specified marker


def show_position(data, marker_index):
    """
    Show the x, y, and z position time history of a given marker.

    Parameters
    ----------
    data : array
        Point (marker) data containing positions (3 or 4 x N x k) where N =
        number of markers, k = number of frames
    marker_index : int
        Index of marker (in data array) for which to plot positions.

    Returns
    -------
    None. Plots marker postions
    """
    # Plot position across frames
    # x position
    plt.subplot(3, 1, 1)
    plt.plot(data[0, marker_index, :])  # plot x position
    # y position
    plt.subplot(3, 1, 2)
    plt.plot(data[1, marker_index, :])  # plot y position
    # z position
    plt.subplot(3, 1, 3)
    plt.plot(data[2, marker_index, :])  # plot z position

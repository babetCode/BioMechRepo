# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:56:49 2023.

@author: Thomas Madden

Inverse dynamics
Perform a standard inverse dynamics routine given motion capture and force
data

v. 5: 09/12/2024
Assumes static data processing is done in Cortex (only track markers for the
portion of the trial in which the participant is motionless).
v. 4: 09/11/2024
Revised to import libraries for lengthy functions.
v. 3: 09/05/2024
Revised to condense code.
v. 2: 08/29/2024
Revised coordinate system for ankle joint complex.

"""
# %% Import libraries

# Standard libraries
from ezc3d import c3d
import matplotlib.pyplot as plt
import scipy.signal as sgl
import numpy as np

# Custom libraries for kinematic/kinetic procedures
import kinetics as kin
import visualization as viz

# %% Clean space
plt.close("all")

# %% Constants

# Specific to trial
p = "A04"  # participant code
fp = 4  # force plate from which to get force data
fs_video = 120  # sampling frequency of video data (Hz)
fs_analog = 480  # sampling frequency of analog (force plate) data (Hz)
static_trial = "Static01"  # name of static trial to analyze
trial = "Fast_01"  # name of dynamic (walking) trial to analyze

# Not specific to trial
# Path to folder containing C3D files
path = (
    "C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/Walking_mechanics/Data/"
)
# Cutoff frequencies
cutoff_markers = 8  # cutoff frequency for filtering marker data (Hz)
# 8Hz is a typical cutoff for walking with a prosthesis, which shows slightly
# higher frequency than nonamputee walking (typically < 6Hz)
cutoff_force = 50  # cutoff frequency for filtering force data (Hz)
threshold = 10  # vertical GRF threshold defining stance phase (N)
# Stance phase is the time while vertical GRF exceeds this threshold.

# %% Read C3D files

# Read static C3D file
filename = path + p + "_C3D/" + p + "_" + static_trial + ".c3d"
c_static = c3d(filename)

# Marker labels
static_marker_labels = c_static["parameters"]["POINT"]["LABELS"]["value"]

# Get data
static_point_data = c_static["data"]["points"]
# 4xNxT (x,y,z,1 x points x frames)
static_analog_data = c_static["data"]["analogs"]
# 1xNxT (value x analogs x frames)

# Read dynamic trial C3D file
filename = path + p + "_C3D/" + p + "_" + trial + ".c3d"
c = c3d(filename)


def get_index(labels, target):
    """
    Get the index of a target given a the list of labels.

    Parameters
    ----------
    labels : list
        List of labels.
    target : string
        Target label for which to get index.

    Returns
    -------
    index : int
        Index of target in the list of labels.

    """
    index = [i for i, label in enumerate(labels) if label == target][0]
    return index


# Get analog data from c3d file
analog_data = c["data"]["analogs"]
labels = c["parameters"]["ANALOG"]["LABELS"]["value"]  # analog channel labels

# Get index corresponding to force plate channel
Fx_index = get_index(labels, "F" + str(fp) + "X")

# Get force plate data
# force data is FX and next 2 channels
# moments data is next 3 channels (3 forces, 3 moments)
F = analog_data[0, Fx_index : Fx_index + 3, :]  # forces (N)
M = analog_data[0, Fx_index + 3 : Fx_index + 6, :]  # moments (N*mm)

# %% Transform force data to lab coordinate frame

# For our lab, the lab and force plate reference frames are as follows:
# Lab frame
x = [1, 0, 0]  # positive anterior
y = [0, 1, 0]  # medial-lateral
z = [0, 0, 1]  # positive vertical
# Force plate frame:
yf = [1, 0, 0]
zf = [0, 0, -1]
xf = np.cross(yf, zf)

# Transform force and moments data to lab frame
# Origins are irrelevant because we are simply rotating the axes
o = np.array([0, 0, 0])
T = kin.transform_matrix(x, y, z, xf, yf, zf, o, o)  # Transformation matrix
R = T[0:3, 0:3]  # rotation matrix
F = np.matmul(R, F)  # transformed force vectors
M = np.matmul(R, M)  # transformed moment vectors

# %% Get center of pressure from force plate data


# %% Filter force data

# Filter force data using a low-pass Butterworth filter
# A fourth-order no-lag filter is equivalent to a second-order filter that
# also filters data in reverse (sosfiltfilt)
sos = sgl.butter(2, cutoff_force, output="sos", fs=fs_analog)
F_filtered = sgl.sosfiltfilt(sos, F, axis=1)  # filter raw force data
M_filtered = sgl.sosfiltfilt(sos, M, axis=1)  # filter raw moments data

# Plot raw and filtered data
plt.plot(F[0, :])  # anterior-posterior ground reaction force
plt.plot(F_filtered[0, :])
plt.plot(F[2, :])  # vertical ground reaction force
plt.plot(F_filtered[2, :])

# Set force data variable to filtered data
F = F_filtered

# %% Remove zero offset

# To account for sets of trials for which the force plates were not zeroed
# properly. Apply to each trial.
t = 1  # number of seconds for which to average data to get offset
F_offset = np.mean(F[:, 0 : t * fs_analog], axis=1)  # mean forces first t s
M_offset = np.mean(M[:, 0 : t * fs_analog], axis=1)  # mean moments first t s
# subtract offset from force and moments data
for i in range(len(F[0])):
    F[:, i] = F[:, i] - F_offset
    M[:, i] = M[:, i] - M_offset

# Plot offset force data
plt.plot(F[0, :])  # anterior-posterior
plt.plot(F[2, :])  # vertical

# %% Get gait events

[ic, to, ms, ps] = kin.get_gait_events(F[0], F[2], threshold)

# Plot with forces to verify events are ID'd correctly
# plot dotted vertical lines at each gait event
y = plt.gca().get_ylim()[1] - 100  # y-coordinate to put text
# 1) initial contact
plt.axvline(x=ic, color="black", linestyle=":", linewidth=1)
plt.text(ic + 5, y, "IC", color="black", fontsize=12)  # label
# 2) midstance
plt.axvline(x=ms, color="black", linestyle=":", linewidth=1)
plt.text(ms + 5, y, "50% stance", color="black", fontsize=12)
# 3) start of propulsion
plt.axvline(x=ps, color="black", linestyle=":", linewidth=1)
plt.text(ps + 5, y - 75, "Propulsion start", color="black", fontsize=12)
# 4) toe-off
plt.axvline(x=to, color="black", linestyle=":", linewidth=1)
plt.text(to + 5, y, "TO", color="black", fontsize=12)

# %% Get marker trajectories

# Get marker labels
labels = c["parameters"]["POINT"]["LABELS"]["value"]  # point (marker) labels
# labels may contain 'Mobility_markerset:' preceding the marker name
label_prefix = "Mobility_markerset:"

# Get marker data
data = c["data"]["points"]  # x, y, z, 1 marker coordinates (global frame)

# %% Filter marker data

# Static data: filter by taking mean positions
static_marker_pos = static_point_data[0:3]

# Filter by taking mean positions of each marker
# Initialize variable containing mean marker x, y, z positions for each frame
static_pos = np.zeros(np.shape(static_marker_pos[:, :, 0]))
for i in range(np.shape(static_pos)[1]):
    pos = static_marker_pos[:, i, :]  # positions of current marker
    # Remove nans
    x = pos[0, ~np.isnan(pos[0])]  # x positions
    y = pos[1, ~np.isnan(pos[1])]  # y positions
    z = pos[2, ~np.isnan(pos[2])]  # z position
    static_pos[:, i] = [
        np.mean(x),
        np.mean(y),
        np.mean(z),
    ]  # mean x, y, z positions

# Marker positions are 0:3 in first dimension (c3d adds a 1 for use with
# transformation matrics)
marker_pos = data[0:3]

# Remove nans: interpolate or replace with 0
# Replace with 0
marker_pos_new = np.nan_to_num(marker_pos, 0)

# Filter marker data using a low-pass Butterworth filter
# A fourth-order no-lag filter is equivalent to a second-order filter that
# also filters data in reverse (sgl.sosfiltfilt)
sos = sgl.butter(2, cutoff_markers, output="sos", fs=fs_video)
marker_pos_filtered = sgl.sosfiltfilt(
    sos, marker_pos_new, axis=2
)  # filter raw data

data = marker_pos_filtered  # set data variable to the filtered data


# %% Plot markers to ensure they are correctly indexed

# Static data
viz.show_markers(static_pos, marker_index=get_index(labels, "LCAL"))
plt.title("Static marker data", fontsize=16)

# Dynamic data
# show_markers(data, frame) # no marker highlighted
k = fs_analog / fs_video  # factor to convert analog frames to video frames
index = get_index(labels, "LCAL")  # idex for LCAL marker
viz.show_markers(
    data, frame=int(ic / k), marker_index=index
)  # highlight LCAL, at initial contact
plt.title("Initial contact", fontsize=16)
viz.show_markers(data, frame=int(ms / k), marker_index=index)  # at midstance
plt.title("Midstance", fontsize=16)
viz.show_markers(
    data, frame=int(ps / k), marker_index=index
)  # at start of propulsion
plt.title("Start of propulsion", fontsize=16)
viz.show_markers(data, frame=int(to / k), marker_index=index)  # at toe-off
plt.title("Toe-off", fontsize=16)

# Plot raw and filtered positions of a specific marker
fig = plt.figure()
marker = "LCAL"  # specify marker
marker_index = get_index(labels, marker)  # index of marker
viz.show_position(marker_pos, marker_index)  # raw positions
viz.show_position(data, marker_index)  # filtered
# Add title, axis labels, legend
fig.suptitle(
    f"{marker} position", fontsize=16
)  # super title (to entire figure)
fig.legend(["Raw", "Filtered"])  # add legend to entire figure
plt.subplot(3, 1, 1)
plt.ylabel("X", fontsize=14)
plt.subplot(3, 1, 2)
plt.ylabel("Y", fontsize=14)
plt.subplot(3, 1, 3)
plt.ylabel("Z", fontsize=14)
plt.xlabel("Video frame", fontsize=14)
plt.tight_layout()  # ensure titles/labels/legends/plots do not overlap

# %% Define local to anatomical transformation matrices using static data

# Define anatomical coordinate systems

# Anatomical markers
# Left knee
l_lfe = static_pos[:, get_index(labels, "LLFE")]  # lateral femoral epicondyle
l_mfe = static_pos[:, get_index(labels, "LMFE")]  # medial femoral epicondyle
# Left ankle
l_lmal = static_pos[:, get_index(labels, "LLMAL")]  # lateral malleolus
l_mmal = static_pos[:, get_index(labels, "LMMAL")]  # medial malleolus
# Left MTPJ
l_mtp5 = static_pos[:, get_index(labels, "LMTP5")]  # head of MTP5
l_mtp1 = static_pos[:, get_index(labels, "LMTP1")]  # head of MTP1
# Right knee
r_lfe = static_pos[:, get_index(labels, "RLFE")]  # lateral femoral epicondyle
r_mfe = static_pos[:, get_index(labels, "RMFE")]  # medial femoral epicondyle
# Right ankle
r_lmal = static_pos[:, get_index(labels, "RLMAL")]  # lateral malleolus
r_mmal = static_pos[:, get_index(labels, "RMMAL")]  # medial malleolus
# Right MTPJ
r_mtp1 = static_pos[:, get_index(labels, "RMTP1")]  # head of MTP1
r_mtp5 = static_pos[:, get_index(labels, "RMTP5")]  # head of MTP5

# Define knee joint center as midpoint between medial and lateral knee markers
r_kjc = 1 / 2 * (r_mfe + r_lfe)  # right knee joint center
l_kjc = 1 / 2 * (l_mfe + l_lfe)  # left knee joint center

# Define ankle joint center as midpoint between MMAL and LMAL
r_ajc = 1 / 2 * (r_mmal + r_lmal)  # right ankle joint center
l_ajc = 1 / 2 * (l_mmal + l_lmal)  # left ankle joint center

# Define MTP joint center as midpoint between MTP5 and MTP1
r_mtpjc = 1 / 2 * (r_mtp5 + r_mtp1)  # right MTP joint center
l_mtpjc = 1 / 2 * (l_mtp5 + l_mtp1)  # left MTP joint center

# Define shank anatomical coordinate system
# origin: ankle joint center
# Left
l_shank_o = l_ajc
# Right
r_shank_o = r_ajc
# z-axis: line from ankle joint center to knee joint center, normalized
# Left
l_shank_z = (l_kjc - l_ajc) / np.linalg.norm(l_kjc - l_ajc)
# Right
r_shank_z = (r_kjc - r_ajc) / np.linalg.norm(r_kjc - r_ajc)
# x-axis: cross product of line from LMAL to MMAL and z-axis, normalized
# Left
l_shank_x = np.cross(l_mmal - l_lmal, l_shank_z)
l_shank_x = l_shank_x / np.linalg.norm(l_shank_x)  # normalize
# Right
r_shank_x = np.cross(r_mmal - r_lmal, r_shank_z)
r_shank_x = r_shank_x / np.linalg.norm(r_shank_x)  # normalize
# y-axis: cross product of z-axis and x-axis
# Left
l_shank_y = np.cross(l_shank_z, l_shank_x)
# Right
r_shank_y = np.cross(r_shank_z, r_shank_x)

# Define foot anatomical coordinate system
# origin: ankle joint center
# Left
l_foot_o = l_ajc
# Right
r_foot_o = r_ajc
# z-axis: shank z-axis
# Left
l_foot_z = l_shank_z
# Right
r_foot_z = r_shank_z
# x'-axis: line from ankle joint center to MTP joint center
# Left
l_foot_x = l_mtpjc - l_ajc
# Right
r_foot_x = r_mtpjc - r_ajc
# y-axis: cross product of z-axis and x'-axis, normalized
# Left
l_foot_y = np.cross(l_foot_z, l_foot_x)
l_foot_y = l_foot_y / np.linalg.norm(l_foot_y)  # normalize
# Right
r_foot_y = np.cross(r_foot_z, r_foot_x)
r_foot_y = r_foot_y / np.linalg.norm(r_foot_y)  # normalize
# x-axis: cross product of y and z axes
# Left
l_foot_x = np.cross(l_foot_y, l_foot_z)
# Right
r_foot_x = np.cross(r_foot_y, r_foot_z)

# Deine local coordinate systems

# Tracking markers
# Left shank
l_sh1 = static_pos[:, get_index(labels, "LSH1")]  # left shank cluster
l_sh2 = static_pos[:, get_index(labels, "LSH2")]
l_sh3 = static_pos[:, get_index(labels, "LSH3")]
l_sh4 = static_pos[:, get_index(labels, "LSH4")]
# Left foot
l_cal = static_pos[:, get_index(labels, "LCAL")]  # left calcaneus
# Right shank
r_sh1 = static_pos[:, get_index(labels, "RSH1")]  # right shank cluster
r_sh2 = static_pos[:, get_index(labels, "RSH2")]
r_sh3 = static_pos[:, get_index(labels, "RSH3")]
r_sh4 = static_pos[:, get_index(labels, "RSH4")]
# Right foot
r_cal = static_pos[:, get_index(labels, "RCAL")]  # right calcaneus

# Define foot local coordinate system
# origin: CAL
# Left
l_foot_lcs_o = l_cal
# Right
r_foot_lcs_o = r_cal
# x-axis: CAL to MTP joint center, normalized
# Left
l_foot_lcs_x = l_mtpjc - l_cal
l_foot_lcs_x = l_foot_lcs_x / np.linalg.norm(l_foot_lcs_x)  # normalize
# Right
r_foot_lcs_x = r_mtpjc - r_cal
r_foot_lcs_x = r_foot_lcs_x / np.linalg.norm(r_foot_lcs_x)  # normalize
# z-axis: cross product of x-axis and line from MTP5 to MTP1, normalized
# Left
l_foot_lcs_z = np.cross(l_foot_lcs_x, l_mtp1 - l_mtp5)
l_foot_lcs_z = l_foot_lcs_z / np.linalg.norm(l_foot_lcs_z)  # normalize
# Right
r_foot_lcs_z = np.cross(r_foot_lcs_x, r_mtp1 - r_mtp5)
r_foot_lcs_z = r_foot_lcs_z / np.linalg.norm(r_foot_lcs_z)  # normalize
# y-axis: cross product of z-axis and x-axis
# Left
l_foot_lcs_y = np.cross(l_foot_lcs_z, l_foot_lcs_x)
# Right
r_foot_lcs_y = np.cross(r_foot_lcs_z, r_foot_lcs_x)

# Define shank local coordinate system
# origin: SH1
# Left
l_shank_lcs_o = l_sh1
# Right
r_shank_lcs_o = r_sh1
# x-axis: SH1 to SH3, normalized
# Left
l_shank_lcs_x = (l_sh3 - l_sh1) / np.linalg.norm(l_sh3 - l_sh1)
# Right
r_shank_lcs_x = (r_sh3 - r_sh1) / np.linalg.norm(r_sh3 - r_sh1)
# y-axis: cross product of line from SH2 to SH1 and x-axis, normalized
# Left
l_shank_lcs_y = np.cross(l_sh1 - l_sh2, l_shank_lcs_x)
l_shank_lcs_y = l_shank_lcs_y / np.linalg.norm(l_shank_lcs_y)  # normalize
# Right
r_shank_lcs_y = np.cross(r_sh1 - r_sh2, r_shank_lcs_x)
r_shank_lcs_y = r_shank_lcs_y / np.linalg.norm(r_shank_lcs_y)  # normalize
# z-axis: cross product of x-axis and y-axis
# Left
l_shank_lcs_z = np.cross(l_shank_lcs_x, l_shank_lcs_y)
# Right
r_shank_lcs_z = np.cross(r_shank_lcs_x, r_shank_lcs_y)

# Define transformation matrices

# Transformation matrix: foot local to anatomical coordinate system
# Left
T_lfoot_lcs_acs = kin.transform_matrix(
    l_foot_x,
    l_foot_y,
    l_foot_z,
    l_foot_lcs_x,
    l_foot_lcs_y,
    l_foot_lcs_z,
    l_foot_o,
    l_foot_lcs_o,
)
# Right
T_rfoot_lcs_acs = kin.transform_matrix(
    r_foot_x,
    r_foot_y,
    r_foot_z,
    r_foot_lcs_x,
    r_foot_lcs_y,
    r_foot_lcs_z,
    r_foot_o,
    r_foot_lcs_o,
)
# Transformation matrix: shank local to anatomical coordinate system
T_lshank_lcs_acs = kin.transform_matrix(
    l_shank_x,
    l_shank_y,
    l_shank_z,
    l_shank_lcs_x,
    l_shank_lcs_y,
    l_shank_lcs_z,
    l_shank_o,
    l_shank_lcs_o,
)
T_rshank_lcs_acs = kin.transform_matrix(
    r_shank_x,
    r_shank_y,
    r_shank_z,
    r_shank_lcs_x,
    r_shank_lcs_y,
    r_shank_lcs_z,
    r_shank_o,
    r_shank_lcs_o,
)

# %% Kinematics and kinetics

# Static trial

# Get segment and joint angles
# foot angle
# ankle angle
# transformation matrix: foot anatomical to shank anatomical coordinate system
T_lfoot_lshank = 


# Dynamic trial

# Get segment and joint angles for each video frame

# Get foot angle

# Get ankle angle
# transformation matrix: foot anatomical to shank anatomical coordinate system

# Get ground reaction force center of pressure

# Get ground reaction forces and moments

# Get ankle joint reaction forces

# Get ankle moment

# Get ankle power
# 1DOF, 3DOF (rotational) ankle power

# 6DOF ankle (rotational + translational) power

# Distal shank power
# shank COM (in global coordinate system)
# shank COM velocity (in global coordinate system)
# vector from shank COM to GRF COP (global coordinate system)
# shank rotational velocity (global coordinate system)

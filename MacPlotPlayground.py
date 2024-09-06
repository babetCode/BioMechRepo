from math import *
from ezc3d import c3d
import matplotlib.pyplot as plt
import scipy.signal as sgl
from scipy.fft import fft
import numpy as np
import pandas as pd

print('program starting')
# Give participant, speed, trial, and leg (left: 'L' or right: 'R')
p = "C07"  # participant
speed = "Fast"  # walking speed
trial = "07"
leg = "L"

fp = 4  # force plate from which to get force data

# Get file path
filename = (
    p + "_C3D/" + p + "_" + speed + "_" + trial + ".c3d"
)  # name of file corresponding to the trial to be analyzed
path = (
    "/Users/adrienbabet/Documents/490R/Walking C3D files/"
)
filepath = path + filename  # resulting file path

c = c3d(filepath)  # load C3D file

print('c: ', c)
print('0 row: ', c['data']['analogs'][0][0])

point_data = c["data"]["points"]

analog_data = c["data"]["analogs"]

fs = 1000  # sampling frequency of analog data

# %% Clean space
plt.close("all")

# Get force plate channel
""" Force plate analog channels:
    X: lateral-medial, Y: posterior-anterior, Z: downward vertical
    index 0 = channel 1, etc.
"""
fp_label = (
    "F" + str(fp) + "Y"
)  # concatenate to string (Y is posterior-anterior direction)
labels = c["parameters"]["ANALOG"]["LABELS"]["value"]  # analog channel labels
for i in range(len(labels)):
    if (
        labels[i] == fp_label
    ):  # find label for specified force plate Y-direction (posterior-anterior)
        index = i  # set index to that of specified force plate
        break

Fx_raw = analog_data[0, index, :]

print('FX RAW: ', Fx_raw)

plt.plot(Fx_raw)

""" IMU placement:
    1 left upper thigh
    2 left lower thigh
    3 left upper shank
    4 left lower shank
    5 left foot
    6 righ upper thigh
    7 right lower thigh
    8 right upper shank
    9 right lower shank
    10 right foot
    11 sacrum
"""
# %% Get IMU data

# IMU sensor labels
IMU_labels = {
    "L_distal_thigh": 2,
    "L_distal_shank": 4,
    "L_foot": 5,
    "R_distal_thigh": 7,
    "R_distal_shank": 9,
    "R_foot": 10,
    "Sacrum": 11,
}
# %%
# Analog data label
base_label = "DelsysTrignoBase 1: Sensor "  # base label of IMU analog data
acc_label = "IM ACC"  # part of label corresponding to IMU accelerations
sensors = [
    str(IMU_labels["L_distal_shank"]),
    str(IMU_labels["R_distal_shank"]),
    str(IMU_labels["Sacrum"]),
]  # sensors analyzed
sensor_labels = [None] * len(sensors) * 3  # preallocate sensor labels array
# (3 axes per sensor)
indices = np.zeros(len(sensors) * 3)  # preallocate array containing indices

# %%
# of analog data corresponding to the sensors
labels = c["parameters"]["ANALOG"]["LABELS"]["value"]  # labels of analog data

print(labels[32])

# Get labels
for i in range(len(sensors)):
    # Get labels
    sensor_labels[i * 3] = base_label + sensors[i] + acc_label + " Pitch"
    sensor_labels[i * 3 + 1] = base_label + sensors[i] + acc_label + " Roll"
    sensor_labels[i * 3 + 2] = base_label + sensors[i] + acc_label + " Yaw"
    # Get indices
for i in range(0, len(sensor_labels), 3):  # sensors are in groups of 3
    for j in range(len(labels)):
        if labels[j] == sensor_labels[i]:  # find sensor label in analog data
            indices[i] = j  # set corresponding index
            indices[i + 1] = j + 1
            indices[i + 2] = j + 2

# Convert indices to inegers
indices = np.int_(indices)

# L distal shank resultant acceleration
L_shank_a_ml = analog_data[0, indices[0], :]  # medio-lateral ("pitch") axis
L_shank_a_v = analog_data[0, indices[1], :]  # vertical ("roll") axis
L_shank_a_ap = analog_data[0, indices[2], :]  # antero-posterior ("yaw") axis
L_shank_a = np.sqrt(L_shank_a_ml**2 + L_shank_a_v**2 + L_shank_a_ap**2)

# R distal shank resultant acceleration (sensor 9)
R_shank_a_ml = analog_data[0, indices[3], :]
R_shank_a_v = analog_data[0, indices[4], :]
R_shank_a_ap = analog_data[0, indices[5], :]
R_shank_a = np.sqrt(R_shank_a_ml**2 + R_shank_a_v**2 + R_shank_a_ap**2)

# sacrum resultant acceleration
sacrum_a_ml = -1 * analog_data[0, indices[6], :]
sacrum_a_v = -1 * analog_data[0, indices[7], :]
sacrum_a_ap = analog_data[0, indices[8], :]
sacrum_a = np.sqrt(sacrum_a_ml**2 + sacrum_a_v**2 + sacrum_a_ap**2)

# Plot L shank accelerations
plt.figure()
plt.plot(L_shank_a_ml, label="medio-lateral")
plt.plot(L_shank_a_v, label="vertical")
plt.plot(L_shank_a_ap, label="antero-posterior")
plt.plot(L_shank_a, label="resultant")
# overlay force
#plt.plot(Fx / Fx_max, label="AP force")
plt.legend()
# add title
plt.title("L shank accelerations with AP force")

# Plot R shank accelerations
plt.figure()
plt.plot(R_shank_a_ml, label="medio-lateral")
plt.plot(R_shank_a_v, label="vertical")
plt.plot(R_shank_a_ap, label="antero-posterior")
plt.plot(R_shank_a, label="resultant")
# overlay force
#plt.plot(Fx / Fx_max, label="AP force")
plt.legend()
# add title
plt.title("R shank accelerations with AP force")

# Plot sacrum accelerations
plt.figure()
plt.plot(sacrum_a_ml, label="medio-lateral")
plt.plot(sacrum_a_v, label="vertical")
plt.plot(sacrum_a_ap, label="antero-posterior")
plt.plot(sacrum_a, label="resultant")
# overlay force
#plt.plot(Fx / Fx_max, label="AP force")
plt.legend()
# add title
plt.title("Sacral accelerations with AP force")

plt.show(block=False)
#plt.pause(0.001) # Pause for interval seconds.
input("Press [enter] to close plots.")
plt.close('all') # all open plots are correctly closed after each run
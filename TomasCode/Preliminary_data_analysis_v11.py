# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:56:49 2023
@author: tm4dd

v11: 07/05/2024
Revised to account for double force peaks in C10
v9: 06/27/2024
Revised to account for peak forces lower than the impact on the force plate
v8: 06/26/2024
Revised to account for lost force data in C08
v7: 06/26/2024
No changes made to this file
v6: 06/26/2024
Revised to save data into csv file (one file per participant)
v5: 06/20/2024
Revised to account for different channels for IMU data
v4: 06/18/2024
Revised to account for different channels for force data
Still need to do this for IMU data
v3: 06/15/2024
Revised to account for missing samples
v2: 03/07/2024
Revised for participant A01 (Force recorded at 500 Hz, half samples missing)

Preliminary data analysis
Extracts peak propulsive (anteior ground reaction) force and
sacral and shank acceleration near toe-off

"""
# %% Import libraries
from ezc3d import c3d
import matplotlib.pyplot as plt
import scipy.signal as sgl
from scipy.fft import fft
import numpy as np
import pandas as pd

# %% Read dynamic trial C3D file

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
    'C:/Users/goper/Files/vsCode/490R/Walking_C3D_files/'
)
filepath = path + filename  # resulting file path

c = c3d(filepath)  # load C3D file

point_data = c["data"]["points"]
analog_data = c["data"]["analogs"]

fs = 1000  # sampling frequency of analog data

# %% Clean space
plt.close("all")

# %% Get force data

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

plt.plot(Fx_raw)

""
# %% Correct time points of force data
# Move samples to actual frame numbers and interpolate between frames to
# upsample the raw force data. Multiply frame numbers by the factor k by which
# the number of force samples is off from the number of expected force samples.

n_expected = len(Fx_raw)  # expected number of samples
print(f"expected samples = {n_expected}")
# To get the actual (recorded) number of samples, find the point at which force
# becomes constantly zero
n_actual = 0  # actual number of samples
for i in range(len(Fx_raw)):
    # check if the rest of the samples to the end are zero
    if np.all(Fx_raw[i : len(Fx_raw)] == np.zeros(len(Fx_raw) - i)):
        n_actual = i
        print(f"actual samples = {n_actual}")
        break

if n_actual > 0:  # if there are missing samples
    k = n_expected / n_actual  # factor = expected samples / actual samples
    print(f"expected / actual samples = {k}")
    # k = int(k)  # convert to integer

    # Correct force data by moving samples to their expected frame numbers and
    # interpolating between samples
    F_corrected = np.zeros(len(Fx_raw))  # corrected force
    F_corrected[0] = Fx_raw[0]
    for i in range(1, n_actual):
        F_corrected[int(k * i)] = Fx_raw[
            i
        ]  # move samples to actual frames (multiply by factor)
        # plt.plot(F_corrected)

    # Interpolate force data: replace zeros with interpolation
    for i in range(1, len(F_corrected) - 1):
        if F_corrected[i] == 0:
            # interpolate:
            slope = (
                F_corrected[i + int(np.ceil(k)) - 1] - F_corrected[i - 1]
            ) / k
            for j in range(i, i + int(np.ceil(k)) - 1):
                F_corrected[j] = F_corrected[i - 1] + slope * (j + k - 1 - i)
    plt.plot(F_corrected)

    # To work with the following code, set F_raw to F_corrected
    Fx_raw = F_corrected

# %% Spectral analysis of force data (to help choose cutoff frequency)
# Get portion of force data to analyze (approximate)
"""
start = 5600
end = 6600
y = Fx_raw[start:end]  # portion of force data to analyze

# Verify using sine waves
# x = np.linspace(0,4,fs*4 + 1) # time in seconds
# f = 5 # frequency in Hz
# y1 = np.sin(2 * np.pi * f * x)
# f = 3
# y2 = 2*np.sin(2 * np.pi * f * x)
# y = y1 + y2 # sum sine waves
# plt.figure()
# plt.plot(x,y)

n = len(y)  # sample size
f = np.array(range(n)) * fs / n  # frequency range
yf = fft(y)  # Fourier transform of force data
power = np.abs(yf) ** 2 / n  # power of signal
power_relative = power / sum(power[0 : int(n / 2)])  # power relative to total
plt.figure()
plt.plot(
    f[0:60], power_relative[0:60]
)  # plot power vs. frequency (low frequency)
"""
# %% Filter force data using a low-pass Butterworth filter
plt.figure()

F_cutoff = 50  # cutoff frequency (Hz)
sos = sgl.butter(2, F_cutoff, output="sos", fs=fs)  # 4th order (recursive)
Fx = sgl.sosfiltfilt(sos, Fx_raw, 0)  # filter raw Fx data (recursive)

# Plot raw and filtered force data
plt.plot(Fx_raw)
plt.plot(Fx)
plt.title("AP force")

# %% Correct force for zero offset
# offset = average of non-force measurement
offset = np.mean(Fx[0 : 2 * fs])  # average of first two seconds
# subtract offset from force data
Fx = Fx - offset

# Plot offset data
plt.plot(Fx)

# %% Account for forces lower than impact peaks on force plate
# For A05

# First filter out the impact
F_cutoff = 15
sos = sgl.butter(2, F_cutoff, output="sos", fs=fs)
Fx_new = sgl.sosfiltfilt(sos, Fx, 0)
# Plot result
plt.plot(Fx_new)

# Get peak force after filtering out impact
Fx_new_max = max(Fx_new)  # peak force with impact filtered out
print(f"Fx_new_max = {Fx_new_max}")
Fx_new_max_index = np.argmax(Fx_new)  # index of peak with impact filtered out
print(f"This occurs at {Fx_new_max_index}")
# Get max near max when impact is filtered out
Fx_near_max = Fx[Fx_new_max_index - 50 : Fx_new_max_index + 50]

# %% Get max force
if p == "C10" and speed == "Fast" and (trial == "04" or trial == "03"):
    Fx = Fx[1:8000]
Fx_max = max(Fx)
Fx_max_index = np.argmax(Fx)
if Fx_max > Fx_new_max:  # for the case where impact > peak propulsive force
    Fx_max = max(Fx_near_max)  # get max near max when impact is filtered out
    Fx_max_index = np.argmax(Fx_near_max) + Fx_new_max_index - 50
print(f"Fx_max = {Fx_max}")
print(f"Fx_max occurs at {Fx_max_index}")

# %% Determine window of search for peak accelerations
i = Fx_max_index  # initialize index at index of peak force
f = Fx[i]  # force at index
while f > 0:  # search for the point of zero crossing of force
    i = i - 1  # subtract 1 from i to look backwards
    f = Fx[i]
# search from zero crossing of force to 0.25s after max force
start = i
end = Fx_max_index + int(0.25 * fs)
print(f"start = {start}")
print(f"end = {end}")

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
plt.plot(Fx / Fx_max, label="AP force")
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
plt.plot(Fx / Fx_max, label="AP force")
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
plt.plot(Fx / Fx_max, label="AP force")
plt.legend()
# add title
plt.title("Sacral accelerations with AP force")

# %% Get max accelerations within search window

# Sacrum AP
sacrum_a_ap_window = sacrum_a_ap[start:end]
max_sacrum_a_ap = max(sacrum_a_ap_window)
print(f"Max sacrum AP acceleration = {max_sacrum_a_ap}")

# Sacrum resultant
sacrum_a_window = sacrum_a[start:end]
max_sacrum_a = max(sacrum_a_window)
print(f"Max sacrum resultant acceleration = {max_sacrum_a}")

# L Shank resultant
L_shank_a_window = L_shank_a[start:end]
max_L_shank_a = max(L_shank_a_window)
print(f"Max L shank resultant acceleration = {max_L_shank_a}")

# R shank resultant
R_shank_a_window = R_shank_a[start:end]
max_R_shank_a = max(R_shank_a_window)
print(f"Max R shank resultant acceleration = {max_R_shank_a}")

# %% Save data to csv file

file = path + "/" + p + "_data.csv"  # file to read and write to
# contains the data for the current participant

# Read in data (existing participant) or start a new data table
file_exists = True  # variable indicating whether the file exists
try:
    data = pd.read_csv(file)  # read in data
except:  # write new file if the file does not exist
    df = pd.DataFrame()  # create empty data frame
    data = df.to_csv(file, index=False)  # create empty csv file
    file_exists = False  # indicate that file did not exist

# Set up new row of data
new_row_trial = np.array([int(trial)])
new_row_str = np.array([speed, leg])  # string variables
new_row_num = np.array(
    [Fx_max, max_sacrum_a, max_sacrum_a_ap]
)  # numerical variables
if leg == "L":  # left side
    new_row_num = np.append(new_row_num, max_L_shank_a)
else:  # right side
    new_row_num = np.append(new_row_num, max_R_shank_a)
# Make row vectors
new_row_trial = new_row_trial[np.newaxis, :]
new_row_str = new_row_str[np.newaxis, :]
new_row_num = new_row_num[np.newaxis, :]

# Make data frame
new_row_trial = pd.DataFrame(new_row_trial, columns=["Trial"])
new_row_str = pd.DataFrame(new_row_str, columns=["Speed", "Leg"])
new_row_num = pd.DataFrame(
    new_row_num,
    columns=[
        "Peak_Fp (N)",
        "Peak_sacral_acc (g)",
        "Peak_sacral_acc_AP (g)",
        "Peak_shank_acc (g)",
    ],
)

new_row = pd.concat(
    [new_row_trial, new_row_str, new_row_num], axis=1
)  # concatenate to single row

# Add new row to data
# Handle duplicate rows
if file_exists:  # if the file existed (not empty)
    # if trial, speed, and side are all the same for any existing row of data,
    # do not save duplicate data
    duplicate = False  # variable indicating whether data is duplicate
    for i in range(len(data)):  # for every row of existing data
        last_row = data.iloc[i]  # row of existing data frame
        # Check if new row is duplicate
        if (
            new_row["Trial"][0] == last_row["Trial"]
            and new_row["Speed"][0] == last_row["Speed"]
            and new_row["Leg"][0] == last_row["Leg"]
        ):
            duplicate = True  # indicate that the row is a duplicate
            print("duplicate")
            break
    if not duplicate:  # not duplicate; add row to data frame
        data = pd.concat(
            [data, new_row], ignore_index=True
        )  # add new row to data
else:
    data = new_row  # if data is empty, set equal to the new row

# Write to file
data.to_csv(file, index=False)

# %% Make nice plots for figures

# get time
# force_time = np.linspace(0, len(Fx) / fs, len(Fx))  # time for force

# # %% Save force and acceleration to file to make figures in Matlab
# new_shape = (len(Fx), 1)  # new shape to make columns
# time = np.reshape(force_time, new_shape)  # make time column vector
# force = np.reshape(Fx, new_shape)  # make force column vector
# sacrum_a = np.reshape(sacrum_a, new_shape)  # make sacrum acceleration column
# shank_a = np.reshape(L_shank_a, new_shape)  # make shank acceleration column
# data = np.column_stack((time, force, sacrum_a, shank_a))  # make into matrix
# np.savetxt("Force_and_accelerations.csv", data, delimiter=",")  # save file

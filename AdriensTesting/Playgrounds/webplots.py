import sys
import numpy as np
import plotly.graph_objects as go
sys.path.insert(0, 'c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo')
from imufunctions import*

# Load data
mypath = adrien_c3d_path()
df = c3d_analogs_df('C07', 'Fast', '07', mypath)

# Function to get IMU data
def getimus():
    imuplacements = ['L_prox_thigh', 'L_dist_thigh', 'L_prox_shank', 'L_dist_shank', 'L_foot', 
                     'R_prox_thigh', 'R_dist_thigh', 'R_prox_shank', 'R_dist_shank', 'R_foot', 'sacrum']
    allimus = []
    for i in range(11):
        allimus.append(imu(imuplacements[i], df, i+1))
    return allimus

# Get IMUs and prepare plot data
imus = getimus()
tobeplotted = [i+1 for i in range(11)]

# Create a Plotly figure
fig = go.Figure()

# Plot IMU accelerometer data (pitch, roll, yaw)
for i in tobeplotted:
    for j, pry in enumerate(['pitch', 'roll', 'yaw']):
        fig.add_trace(go.Scatter(
            y=9 * imus[i-1].acc_data.iloc[j],
            mode='lines',
            name=f'{imus[i-1]} ACC {pry}'
        ))

# Plot gravity line
gravity = np.ones(imus[0].frames) * -9.80665
fig.add_trace(go.Scatter(
    y=gravity,
    mode='lines',
    name='gravity'
))

# Update layout and show the plot
fig.update_layout(
    title='IMU Data with Gravity',
    xaxis_title='time (ms)',
    yaxis_title='acc (m/s^2)',
    legend_title='Legend',
    hovermode='x'
)

fig.show()

"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
Plots                                                                  |
_______________________________________________________________________|
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
Author: Adrien Babet    GitHub: @babetcode    Email: adrienbabet1@gmail.com   |
______________________________________________________________________________|

deprecated
"""

# Get imu functions
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
import imufunctions as myfuncs

# Get C3D path
c3d_fp = myfuncs.c3d_file('C07', 'Fast', '07', myfuncs.adrien_c3d_folder(mydir))
df = myfuncs.c3d_analogs_df(c3d_fp)

def getimus():
    imuplacements = ['L_prox_thigh', 'L_dist_thigh', 'L_prox_shank', 'L_dist_shank', 'L_foot', 'R_prox_thigh', 'R_dist_thigh', 'R_prox_shank', 'R_dist_shank', 'R_foot', 'sacrum']
    allimus = []
    for i in range(11):
        allimus.append(myfuncs.imu(imuplacements[i], df, i+1))
    return allimus

imus = getimus()

tobeplotted = [9, 10]

for i in tobeplotted:
    for j, pry in enumerate(['pitch', 'roll', 'yaw']):
        plt.plot(9*imus[i-1].acc_data.iloc[j], label = f'{imus[i-1]}ACC{pry}')

gravity = np.ones(imus[0].frames) * -9.80665
plt.plot(gravity, label = 'gravity')

plt.legend()
plt.show(block=False)
keepfigure = input('ENTER to close plot')
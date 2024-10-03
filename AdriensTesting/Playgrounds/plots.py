import adrienscomputers
adrienscomputers.adriensdirectory()
from imufunctions import*

mypath = adrien_c3d_path()
df = c3d_analogs_df('C07', 'Fast', '07', mypath)

def getimus():
    imuplacements = ['L_prox_thigh', 'L_dist_thigh', 'L_prox_shank', 'L_dist_shank', 'L_foot', 'R_prox_thigh', 'R_dist_thigh', 'R_prox_shank', 'R_dist_shank', 'R_foot', 'sacrum']
    allimus = []
    for i in range(11):
        allimus.append(imu(imuplacements[i], df, i+1))
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
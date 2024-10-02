import sys
sys.path.insert(0, 'c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo')
from imufunctions import*

mypath = adrien_c3d_path()
df = c3d_analogs_df('C07', 'Fast', '07', mypath)
rdshank = imu('right_distal_shank', df, 9)
rfoot = imu('right_foot', df, 10)
gravity = np.ones(rdshank.frames)

plt.plot(gravity, label='gravity')
plt.plot(rdshank.net_acc)
plt.plot(rfoot.net_acc)
plt.show()
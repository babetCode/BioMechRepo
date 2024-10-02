import sys
sys.path.insert(0, 'c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo')
from imufunctions import*

mypath = adrienC3Dpath()
df = c3d_analogs_df('C07', 'Fast', '07', mypath)
print(df)
rdshank = imu('right_distal_shank', df, 9)
rfoot = imu('right_foot', df, 10)
gravity = np.ones(rdshank.frames)
plt.close('all')
plt.plot(gravity, label='gravity')
plt.plot(rdshank.net_acc, label='rdshank')
plt.plot(rfoot.net_acc, label='rfoot')
plt.legend()
plt.show(block=False)
keepfigure = input('ENTER to close plot')
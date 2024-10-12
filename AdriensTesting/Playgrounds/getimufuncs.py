#get imufunctions
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
from imufunctions import *

df = c3d_analogs_df('C07', 'Fast', '07', adrien_c3d_path(mydir))

print(df)

# my3dplot = plt.figure().add_subplot(projection='3d')

# my3dplot.set_xlabel('x')

# my3dplot.set_ylabel('y')

# my3dplot.set_zlabel('z')

# plot3axes(my3dplot)

# plt.show()

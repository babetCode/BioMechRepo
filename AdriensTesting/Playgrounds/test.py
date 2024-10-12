#get imufunctions
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
from imufunctions import *

c3dpath = adrien_c3d_path(mydir)

print(c3dpath)

# my3dplot = plt.figure().add_subplot(projection='3d')

# my3dplot.set_xlabel('x')

# my3dplot.set_ylabel('y')

# my3dplot.set_zlabel('z')

# plot3axes(my3dplot)

# plt.show()

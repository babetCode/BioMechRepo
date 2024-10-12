#get imufunctions
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
from imufunctions import *

#get c3d path
c3dpath = adrien_c3d_path(mydir)
print(c3dpath)
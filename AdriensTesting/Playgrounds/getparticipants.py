# get imufunctions
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
from imufunctions import *
import glob

# participants = [filepath.rsplit('\\', 1)[-1][:3] for filepath in glob.glob(f'{adrien_c3d_path(mydir)}/*')]

# print(participants)

pars = []

tris = []

for n, filepath in enumerate(glob.glob(f'{adrien_c3d_path(mydir)}/*')):
    pars.append(filepath.rsplit('\\', 1)[-1])
    tri = []
    for folder in glob.glob(f'{filepath}/*'):
        tri.append(folder.rsplit('\\', 1)[-1])
    tris.append(tri)

print(tris)



# participants = ['A0'+str(i+1) for i in range(6)]+['C0'+str(i) for i in range(3,9)]+['C10','C11','C12','C15','C17']
# print(participants)




"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Convert c3d files to .trc and .mot for OpenSim                                         
________________________________________________________________________

‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Author: Adrien Babet | GitHub: @babetcode | Email: adrienbabet1@gmail.com  
_______________________________________________________________________________
"""

# Get imu functions
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
import imufunctions as myfns
import ezc3d
import numpy as np


# Get C3D path
c3d_fp = myfns.c3d_file('C07', 'Fast', '07', myfns.adrien_c3d_folder(mydir))

trc_file = 'C:\\Users\\goper\\Files\\vsCode\\490R\\adaptedOsim\\output1.trc'
mot_file = 'C:\\Users\\goper\\Files\\vsCode\\490R\\adaptedOsim\\output1.sto'

myfns.convert_c3d_to_opensim(c3d_fp, trc_file, mot_file)
"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
Read c3d Files                                                         |
_______________________________________________________________________|
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
Author: Adrien Babet    GitHub: @babetcode    Email: adrienbabet1@gmail.com   |
______________________________________________________________________________|
"""

# Add imufunctions.py dir path to the sys path before importing it.
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
import imufunctions as myfns
import ezc3d
import numpy as np

c3d_fp = myfns.c3d_file('C07', 'Fast', '07', myfns.adrien_c3d_folder(mydir))
myc3d = ezc3d.c3d(c3d_fp)
point_data = myc3d['parameters']['POINT']['UNITS']['value']
# point_data = myc3d['data']['points']
print(point_data)
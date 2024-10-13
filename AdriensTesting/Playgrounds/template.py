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
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
import glob
import pandas as pd
import ezc3d
import numpy as np

pars = []

tris = []

for n, par_folder_path in enumerate(glob.glob(f'{myfns.adrien_c3d_folder(mydir)}/*')):
    participant = par_folder_path.rsplit('\\', 1)[-1][:3]
    pars.append(participant)
    session = []
    for mocap_run in glob.glob(f'{par_folder_path}/*'):
        run_id = mocap_run.rsplit('\\', 1)[-1].removesuffix('.c3d')[4:]
        session.append(run_id)
    tris.append(session)

maxlen = max(len(i) for i in tris)

print(f'max len {maxlen}')

for trial in tris:
    if len(trial) < maxlen:
        for i in range(maxlen - len(trial)):
            trial.append('n/a')


mydata = dict(zip(pars, tris))

df = pd.DataFrame(mydata)

print(df)






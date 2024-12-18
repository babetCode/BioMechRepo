"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Make .csv chart of participants and their trials. Sends output file to
path specified on line 65.
Code Status: Functional
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

paths = []

for n, par_folder_path in enumerate(
    glob.glob(f'{myfns.adrien_c3d_folder(mydir)}/*')):
    participant = par_folder_path.rsplit('/', 1)[-1][:3]
    pars.append(participant)
    session = []
    parpaths = []
    for mocap_run in glob.glob(f'{par_folder_path}/*'):
        parpaths.append(mocap_run.replace('/','/'))
        run_id = mocap_run.rsplit('/', 1)[-1].removesuffix('.c3d')[4:]
        session.append(run_id)
    paths.append(parpaths)
    tris.append(session)

maxlen = max(len(i) for i in tris)

print(f'{myfns.adrien_c3d_folder(mydir)}')

for trial in tris:
    trial.sort()
    if len(trial) < maxlen:
        for i in range(maxlen - len(trial)):
            trial.append('n/a')

# for path in paths:
#     if len(path) < maxlen:
#         for i in range(maxlen - len(path)):
#             path.append('n/a')


mydata = dict(zip(pars, tris))

# fulldata = dict(zip(pars, paths))

df = pd.DataFrame(mydata)

df = df.sort_index(axis=1)

df.to_csv('sortedtrials.csv')
print('made csv')
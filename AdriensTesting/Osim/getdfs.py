"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Itterate through files in 'c3d_files' folder and generate .trc and .mot
files compatible with OpenSim. Output files are sent to 'open_sim' files
folder in same directory as 'c3d_files'.
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
import ezc3d
import numpy as np
import os
import pandas as pd
import glob

c3d_files_folder = myfns.adrien_c3d_folder(mydir)

def get_trials_df():
    participants = []
    paths = []
    for participant_folder in glob.glob(f'{c3d_files_folder}*'):
        participant = participant_folder.replace(
            '\\','/').rsplit('/', 1)[-1][:3]
        participants.append(participant)
        parpaths = []
        for trial in glob.glob(f'{participant_folder}/*'):
            parpaths.append(trial.replace('\\','/'))
        paths.append(parpaths)
    maxlen = max(len(path) for path in paths)
    for path in paths:
        if len(path) < maxlen:
            for i in range(maxlen - len(path)):
             path.append('n/a')
    data = dict(zip(participants, paths))
    return(pd.DataFrame(data))

def make_osim_files():
    trials_df = get_trials_df()
    print(trials_df)
    for col in trials_df:
        print(col)
        for n, path in enumerate(trials_df[col]):
            newfp = path.replace('c3d_files', 'df_files').removesuffix(
                    '.c3d').replace('_C3D', '')
            print(newfp)
            if n == 0:
                print(newfp.rsplit('/', 1)[0])
                os.makedirs(newfp.rsplit('/', 1)[0])
            if path != 'n/a':
                myfns.c3d_analogs_df(path).to_csv(f'{newfp}.csv')


def main():
    make_osim_files()


if __name__ == '__main__':
    main()
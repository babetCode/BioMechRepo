from math import *
from ezc3d import c3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set path to c3d file
# SPECIFIC TO ADRIEN-MINIPC (CUSTOMIZE TO YOUR COMPUTER)
mypath = ("/Users/adrienbabet/Documents/490R/Walking C3D files/")


def c3d_analogs_pd(participant, speed, trial, path):
    filename = (
        participant+'_C3D/'+participant+'_'+speed+'_'+trial+'.c3d')
    path = mypath
    file_path = path+filename
    myc3d = c3d(file_path)
    point_data = myc3d['data']['points']
    analog_data = myc3d['data']['analogs']
    analogs = analog_data[0, :, :]
    analog_labels = myc3d['parameters']['ANALOG']['LABELS']['value']
    df = pd.DataFrame(data=analogs, index=analog_labels)
    return df


def main():
    arr = np.array([[(i+1)*(j+2) for i in range(5)] for j in range(3)])
    rows = ['these', 'are', 'names']
    df = pd.DataFrame(data= arr, index= rows)
    print(df)
    print(df.apply(np.sqrt, axis=1, raw=False))
    

if __name__ == "__main__":
    main()
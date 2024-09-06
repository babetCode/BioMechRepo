from math import *
from ezc3d import c3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set path to c3d file
# SPECIFIC TO ADRIEN-MINIPC (CUSTOMIZE TO YOUR COMPUTER)
mypath = ("C:/Users/goper/Files/vsCode/490R/Walking_C3D_files/")

#list of participants
def list_participants():
    participants = []
    for participant in ['A0'+str(i+1) for i in range(6)]:
        participants.append(participant)
    for participant in ['C0'+str(i+1) for i in range(17)]:
        participants.append(participant)
    for participant in ['C0'+str(i) for i in [1,2,9,13,14,16]]:
        participants.remove(participant)
    return participants


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
    participants = [
        'A01', 'A02', 'A03', 'A04', 'A05', 'A06',
        'C03', 'C04', 'C05', 'C06', 'C07', 'C08',
        'C010', 'C011', 'C012', 'C015', 'C017']
    speeds = ['Fast', 'Slow', 'SSWS', 'Static', 'PWS']
    print(c3d_analogs_pd('C07', 'Fast', '07', mypath))

if __name__ == "__main__":
    main()
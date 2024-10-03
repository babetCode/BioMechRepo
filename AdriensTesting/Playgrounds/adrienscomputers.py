import sys

def adriensdirectory():
    """
    adds directory path for imufunctions on Adriens computers.

    """
    abpcpath = 'c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo'
    abmacpath = '/Users/adrienbabet/Documents/490R/IMU_gait_analysis'
    tmpcpath = 'C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/Python_code'
    pathfinder = [abpcpath, abmacpath, tmpcpath]
    for path in pathfinder:
        if path in str(__file__).replace('\\', '/'):
            sys.path.insert(0, f'{path}')
            break

adriensdirectory()

import imufunctions
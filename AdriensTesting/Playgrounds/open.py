import sys
sys.path.insert(0, 'c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo')


def adrienC3Dpath():
    """
    Gets directory path for C3D files on Adriens computers.

    Returns
    -
    mypath: str
        My file path.
    """
    abpcpath = ('c:\\Users\\goper\\Files\\vsCode\\490R\\VScodeIMUrepo', 'C:/Users/goper/Files/vsCode/490R/Walking_C3D_files/')
    abmacpath = ('/Users/adrienbabet/Documents/490R/IMU_gait_analysis', '/Users/adrienbabet/Documents/490R/Walking C3D files/')
    tmpcpath = ('C:\\Users\\tm4dd\\Documents\\00_MSU\\01_PhD_Research\\Python_code', 'C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/Walking_mechanics/Data/')
    pathfinder = dict([abpcpath, abmacpath, tmpcpath])
    for path in pathfinder.keys():
        if path in str(__file__):
            return(pathfinder[path])
            break

print(adrienC3Dpath())


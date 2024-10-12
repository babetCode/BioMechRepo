import sys

class adriensBioMechDir:
    def __init__(self, foo):
        pathfinder = {'c:/Users/goper/Files/vsCode': 'abpc',
                      '/Users/adrienbabet/Documents/vsCode': 'abmac',
                      'C:/Users/tm4dd/Documents': 'tmlaptop'}
        for key in pathfinder:
            if key in str(__file__).replace('\\', '/'):
                self.machine = pathfinder[key]
                break
    def __str__(self):
        return f'{self.machine}'
    
    def getmyfunctions(self):
        path2functions = {'abpc': 'c:/Users/goper/Files/vsCode/490R/BioMechRepo',
            'abmac': '/Users/adrienbabet/Documents/vsCode/490R/BioMechRepo',
            'tmlaptop': 'C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/Python_code'}
        sys.path.insert(0, f'{path2functions[self.machine]}')
        print(f'added "{path2functions[self.machine]}" to sys path')
        print('ready to import imufunctions')      

import imufunctions
print('import succesful')
mydir = imu(1,1,1)

# def adriensdirectory:
#     """
#     adds directory path for imufunctions on Adriens computers.

#     """
#     abpcpath = 'c:/Users/goper/Files/vsCode/490R/BioMechRepo'
#     abmacpath = '/Users/adrienbabet/Documents/vsCode/490R/BioMechRepo'
#     tmpcpath = 'C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/Python_code'
#     pathfinder = [abpcpath, abmacpath, tmpcpath]
#     for path in pathfinder:
#         if path in str(__file__).replace('\\', '/'):
#             sys.path.insert(0, f'{path}')
#             break

# adriensdirectory()

# import imufunctions
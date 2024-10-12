import sys

class BioMechDir:
    def __init__(self):
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

# myDir = BioMechDir()

# myDir.getmyfunctions()

# from imufunctions import *

# my3dplot = plt.figure().add_subplot(projection='3d')

# my3dplot.set_xlabel('x')

# my3dplot.set_ylabel('y')

# my3dplot.set_zlabel('z')

# plot3axes(my3dplot)

# plt.show()


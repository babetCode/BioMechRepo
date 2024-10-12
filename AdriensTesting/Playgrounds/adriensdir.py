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
    
    def add_imu_func_path(self):
        path2functions = {'abpc': 'c:/Users/goper/Files/vsCode/490R/BioMechRepo',
            'abmac': '/Users/adrienbabet/Documents/vsCode/490R/BioMechRepo',
            'tmlaptop': 'C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/Python_code'}
        sys.path.insert(0, f'{path2functions[self.machine]}')
        return f'{self.machine}'
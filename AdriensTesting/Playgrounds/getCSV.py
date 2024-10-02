import sys
sys.path.insert(0, 'c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo')
from imufunctions import*

mypath = adrien_c3d_path()
df = c3d_analogs_df('C07', 'Fast', '07', mypath)

#df.to_csv('C:/Users/goper/Downloads/trialC07test.csv')
import AdriensTesting.Playgrounds.adriensdir as adriensdir
adriensdir.adriensdirectory()
from imufunctions import*

mypath = adrien_c3d_path()
df = c3d_analogs_df('C07', 'Fast', '07', mypath)

trim = df.iloc[32:].drop('DelsysTrignoBase 1: Sensor '+str(i+1)+'EMG' for i in range(11))
print(trim)

final = trim.iloc[np.r_[18:24, 48:54]]
final = final.round(decimals=5)
print(final)
#final.to_csv('C:/Users/goper/Downloads/7fastWalk.csv', index=False, header=False)
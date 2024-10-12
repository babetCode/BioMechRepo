import opensim as osim

import opensim as osim

# Load your C3D file
c3dFile = 'your_file.c3d'
model = osim.Model()
c3dTool = osim.C3DFileAdapter()
table = c3dTool.read(c3dFile)

# Convert to .trc and .mot format
trcFilename = 'output.trc'
motFilename = 'output.mot'

# Create a TRC file from the C3D data
osim.STOFileAdapter.write(table, trcFilename)

# Convert the table to a time series table
timeSeriesTable = osim.TimeSeriesTable(table)

# Write the TimeSeriesTable to a MOT file
osim.STOFileAdapter.write(timeSeriesTable, motFilename)

print(f"Converted {c3dFile} to {trcFilename} and {motFilename}")

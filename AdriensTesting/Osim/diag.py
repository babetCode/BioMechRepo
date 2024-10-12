# Get imu functions
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
from imufunctions import *
import opensim as osim
import os

# Get C3D path
c3dFile = c3d_file('C07', 'Fast', '07', adrien_c3d_folder(mydir))

# Ensure the file exists
if not os.path.isfile(c3dFile):
    raise FileNotFoundError(f"The file {c3dFile} does not exist.")

# OpenSim C3D tool
c3dTool = osim.C3DFileAdapter()
c3dTool.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)
tables = c3dTool.read(c3dFile)  # Read all data tables

# Extract marker and force data
markerTable = c3dTool.getMarkersTable(tables)
forceTable = c3dTool.getForcesTable(tables)

# Check the types of the tables
print(f"Marker table type: {type(markerTable)}")
print(f"Force table type: {type(forceTable)}")

# Write TRC file using TRCFileAdapter
trcFilename = 'output.trc'
trcAdapter = osim.TRCFileAdapter()
trcAdapter.write(markerTable, trcFilename)

# If forceTable is TimeSeriesTableVec3, flatten it
if isinstance(forceTable, osim.TimeSeriesTableVec3):
    print("Flattening the Vec3 table to a scalar table for MOT file.")
    flatForceTable = osim.TableProcessor(forceTable).process()
    motFilename = 'output.mot'
    osim.STOFileAdapter.write(flatForceTable, motFilename)
    print(f"Converted {c3dFile} to {trcFilename} and {motFilename}")
else:
    print("Force table is not of type Vec3, skipping flattening.")

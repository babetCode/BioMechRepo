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

model = osim.Model()
c3dTool = osim.C3DFileAdapter()
c3dTool.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)
tables = c3dTool.read(c3dFile)  # Read all data tables
markerTable = c3dTool.getMarkersTable(tables)  # Extract marker data
forceTable = c3dTool.getForcesTable(tables)

# Convert to .trc and .mot format
trcFilename = 'output.trc'
motFilename = 'output.mot'

# Write TRC file using TRCFileAdapter
trcAdapter = osim.TRCFileAdapter()
trcAdapter.write(markerTable, trcFilename)

# Manually flatten the Vec3 force data into a scalar table
def flatten_force_table(vec3_table):
    """Flatten Vec3 force table to scalar table for .mot file."""
    flat_table = osim.TimeSeriesTable()
    labels = vec3_table.getColumnLabels()

    for i in range(len(labels)):
        # For each Vec3 column, add 3 scalar columns (X, Y, Z)
        for j in range(3):
            flat_table.appendColumn(f'{labels[i]}_{["X", "Y", "Z"][j]}', osim.Vector())

    # Fill the scalar table with corresponding X, Y, Z data
    for row in range(vec3_table.getNumRows()):
        time = vec3_table.getIndependentColumn()[row]
        row_data = []

        for i in range(len(labels)):
            vec3 = vec3_table.getDependentColumnAtIndex(i)[row]
            row_data.extend([vec3.get(j) for j in range(3)])  # Extract X, Y, Z

        # Append row data to the flattened table
        flat_table.appendRow(time, osim.RowVector(row_data))

    return flat_table

# Flatten the force table and write to .mot
forceFlatTable = flatten_force_table(forceTable)
osim.STOFileAdapter.write(forceFlatTable, motFilename)

print(f"Converted {c3dFile} to {trcFilename} and {motFilename}")
# Get imu functions
from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
from imufunctions import *
import opensim as osim
import os
import ezc3d

# Get C3D path
c3dFile = c3d_file('C07', 'Fast', '07', adrien_c3d_folder(mydir))

def write_trc_file(marker_data, labels, frame_rate, output_file):
    """
    Writes marker data to a .trc file compatible with OpenSim.
    """
    num_markers = marker_data.shape[1]
    num_frames = marker_data.shape[0]

    with open(output_file, 'w') as f:
        # Write header
        f.write('PathFileType\t4\t(X/Y/Z)\t{}\n'.format(output_file))
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write('{:.2f}\t{:.2f}\t{}\t{}\tm\t{:.2f}\t1\t{}\n'.format(
            frame_rate, frame_rate, num_frames, num_markers, frame_rate, num_frames))
        
        # Write marker labels
        f.write('Frame#\tTime\t' + '\t'.join([f'{label}_X\t{label}_Y\t{label}_Z' for label in labels]) + '\n')
        f.write('\n')  # Empty line separating header and data

        # Write data
        for i in range(num_frames):
            time = i / frame_rate
            frame_data = marker_data[i].reshape(-1)
            f.write(f'{i+1}\t{time:.5f}\t' + '\t'.join([f'{x:.5f}' for x in frame_data]) + '\n')

def write_mot_file(analog_data, labels, frame_rate, output_file):
    """
    Writes analog data (forces, joint angles, etc.) to a .mot file compatible with OpenSim.
    """
    num_columns = analog_data.shape[1]
    num_frames = analog_data.shape[0]
    
    with open(output_file, 'w') as f:
        # Write header
        f.write('name {}\n'.format(output_file))
        f.write('datacolumns {}\n'.format(num_columns))
        f.write('datarows {}\n'.format(num_frames))
        f.write('range 0 {:.5f}\n'.format(num_frames / frame_rate))
        f.write('endheader\n')

        # Write column headers
        f.write('time\t' + '\t'.join(labels) + '\n')
        
        # Write data
        for i in range(num_frames):
            time = i / frame_rate
            frame_data = analog_data[i]
            f.write(f'{time:.5f}\t' + '\t'.join([f'{x:.5f}' for x in frame_data]) + '\n')

def convert_c3d_to_opensim(c3d_file, trc_file, mot_file):
    # Load the .c3d file
    c3d = ezc3d.c3d(c3d_file)
    
    # Extract marker data (3D coordinates)
    marker_data = c3d['data']['points']  # shape: (4, n_markers, n_frames)
    marker_data = marker_data[:3, :, :].transpose(2, 1, 0)  # Reshape to (n_frames, n_markers, 3)

    # Extract marker labels
    marker_labels = c3d['parameters']['POINT']['LABELS']['value']

    # Extract frame rate
    frame_rate = c3d['parameters']['POINT']['RATE']['value'][0]
    
    # Write marker data to .trc file
    write_trc_file(marker_data, marker_labels, frame_rate, trc_file)

    # Extract analog data (optional, for motion data if available)
    if 'ANALOG' in c3d['parameters']:
        analog_data = np.array(c3d['data']['analogs']).squeeze().T
        analog_labels = c3d['parameters']['ANALOG']['LABELS']['value']
        
        # Write analog data to .mot file
        write_mot_file(analog_data, analog_labels, frame_rate, mot_file)
    else:
        print("No analog data found in the .c3d file. Skipping .mot file creation.")

# Example usage
trc_file = 'output.trc'
mot_file = 'output.mot'

convert_c3d_to_opensim(c3dFile, trc_file, mot_file)
print("Current working directory:", os.getcwd())


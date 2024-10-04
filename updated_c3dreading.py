
import ezc3d
import numpy as np

# Load your C3D file
c3d = ezc3d.c3d("C:/Users/goper/Files/vsCode/490R/Walking_C3D_files/C07_C3D/C07_Fast_07.c3d")

# Extract marker data
markers = c3d['data']['points']  # Shape: (4, #markers, #frames)
frame_rate = c3d['header']['points']['frame_rate']  # Extracted from the C3D file
marker_names = c3d['parameters']['POINT']['LABELS']['value']  # Extract marker names
marker_units = c3d['parameters']['POINT']['UNITS']['value'][0]

if marker_units == "m":
    markers *= 1000  # Convert to millimeters if data is in meters

def write_trc_file(filename, markers, frame_rate):
    num_frames = markers.shape[2]
    num_markers = markers.shape[1]
    
    with open(filename, 'w') as f:
        # TRC Header
        f.write("PathFileType	4	(X/Y/Z)	{}
".format(filename))
        f.write("DataRate	CameraRate	NumFrames	NumMarkers	Units
")
        f.write("{:.1f}	{:.1f}	{}	{}	mm
".format(frame_rate, frame_rate, num_frames, num_markers))
        
        # Write the time and marker names in header
        f.write("Frame#	Time")
        for marker in marker_names:
            f.write("	{}		".format(marker))  # One tab for X, two for Y and Z columns
        f.write("
")
        
        f.write("		")
        for _ in marker_names:
            f.write("	X	Y	Z")  # X/Y/Z headers for each marker
        f.write("
")
        
        # Write marker data for each frame
        for frame in range(num_frames):
            f.write("{}	{:.5f}".format(frame + 1, frame / frame_rate))  # Frame number and time
            for marker in range(num_markers):
                x, y, z = markers[:3, marker, frame]  # X, Y, Z positions
                f.write("	{:.5f}	{:.5f}	{:.5f}".format(x, y, z))
            f.write("
")

write_trc_file("/mnt/data/updated_output.trc", markers, frame_rate)

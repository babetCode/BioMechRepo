import sys
sys.path.insert(0, 'c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo')
from imufunctions import*

class forceplates:
    """
    For analyzing forceplate data.

    Parameters
    -
    name: string
        Name for the force plate.
    df: dataframe
        dataframe containing forceplate data. Using c3d_analogs_df() is reccomended.

    Attributes
    -
    name: str
        As initialized.
    indices: list
        List of analog labels from df.
    start_row_index: int
        Index of the first row in df which contains data for this IMU.
    all_data: dataframe
        Data from df for this specific IMU.
    acc_data: dataframe
        Acceleration data from all_data.
    net_acc: dataframe
        Net acceleration.
    frames: int
        Number of frames in df.
    gyr_data: dataframe
        Gyroscopic data from df.

    Methods
    -
    raw_orientation()
        Attemts to determine orientation with raw data.
    test()
        pass.

    """
    def __init__(self, name, df, sensor_num):
        self.name = name
        self.indices = [row for row in df.index] # list of analog labels
        self.start_row_index = self.indices.index('DelsysTrignoBase 1: Sensor '+str(sensor_num)+'IM ACC Pitch') # find first row label
        self.all_data = df.iloc[self.start_row_index : self.start_row_index+6] # get dataframe of the 6 rows
        
        self.acc_data = self.all_data.iloc[0:3] # get the first 3 rows of acc data
        sqrt_acc = np.square(self.acc_data) # square of all acc data
        net_acc_sq = sqrt_acc.apply(np.sum, axis=0, raw=True) # sum of P,R,Y acc squares for each frame
        self.net_acc = np.sqrt(net_acc_sq) # net acc for each frame
        self.gyr_data = self.all_data.iloc[3:7] # get the next three rows of gyr data
        self.frames = len(self.gyr_data.columns) # get number of frames (same as legth of rows)

    def raw_orientation(self):
        xyz_axes = np.array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]] for i in range(self.frames)]) # xyz_axes[frame, axis, xyz vector]
        [xaxis, yaxis, zaxis] = [xyz_axes[0][i] for i in range(3)]

        # print([i for i in self.gyr_data.iloc[:,0]]) # first column

        # set up 3d figure
        plt.close('all')
        my3dplot = plt.figure().add_subplot(projection='3d')
        my3dplot.set_xlabel('x')
        my3dplot.set_ylabel('y')
        my3dplot.set_zlabel('z')
        plot3axes(my3dplot)

        for i in range (self.frames):
            # print('loop # '+str(i))
            gyr = np.array(self.gyr_data.iloc[:, i-1]) # get gyr date for previous frame
            initial_axes = xyz_axes[i-1,:,:] # get axes for previous frame
            scaled_axes = [initial_axes[j] * gyr[j] for j in range(3)] # scale axes by component rotation velocity
            total_axis = np.sum(scaled_axes, axis=0) # axis of rotation
            # norm_axis = total_axis/np.linalg.norm(total_axis) # normalized axis - NOT NECESSARY AS rotateQuaternion() already does this
            norm_gyr = sqrt(np.sum(np.square(gyr))) # rotational velocity
            angle_deg = norm_gyr/148
            angle_rad = angle_deg * pi/180 
            rotated_axes = np.array([rotateQuaternion(axisvector, angle_rad, total_axis) for axisvector in initial_axes])
            #print(rotated_axes)
            xyz_axes[i,:,:] = rotated_axes
        my3dplot.plot(xyz_axes[:,0,0], xyz_axes[:,0,1], xyz_axes[:,0,2])
        plt.show()


    def plot_net_acc(self, scale):
        plt.plot(self.net_acc*scale, label = 'net acc '+self.name)
    
    def plot_PRY(self, PRY, scale):
        if 'P' in PRY:
            plt.plot(self.gyr_data.iloc[0]*scale, label= 'gyr pitch '+self.name)
        if 'R' in PRY:
            plt.plot(self.gyr_data.iloc[1]*scale, label= 'gyr roll '+self.name)
        if 'Y' in PRY:
            plt.plot(self.gyr_data.iloc[2]*scale, label= 'gyr yaw '+self.name)

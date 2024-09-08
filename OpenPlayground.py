from functionBuilder import *

def main():
    mypath = adrienC3Dpath()
    df = c3d_analogs_df('C07', 'Fast', '07', mypath)
    plt.close('all')
    LDistalShank = imu('LDistShank', df, 2)

    print(np.zeros(10))

    # test_axis_rotation()
    x1 = [1.,0.,0.]
    plot_rotation(x1, pi/2, [0,1,0]) # rotate x=1 around y-axis
    plt.figure()
    LDistalShank.plot_net_acc(150)
    # print(p2) 
    # p3 = rotateQuaternion(p2, pi/2, [1,0,0])
    # print(p3)
    # qp = rotateQuaternion(x1, sqrt(2*(pi/2)*(pi/2)), [sqrt(2)/2,sqrt(2)/2,0])
    # print(qp)
    # plt.figure()
    # LDistalShank.plot_net_acc(150)
    # LDistalShank.plot_PRY('PRY', 1)
    # plt.legend()
    plt.show(block=False)
    close_plots = input('[enter] to close plots >')

if __name__ == "__main__":
    main()
from AdriensFunctions import *

def main():
    mypath = adrienC3Dpath()
    df = c3d_analogs_df('C07', 'Fast', '07', mypath)
    plt.close('all')
    LDistalShank = imu('LDistShank', df, 2)

    LDistalShank.raw_orientation()    

if __name__ == "__main__":
    main()

# this is a test
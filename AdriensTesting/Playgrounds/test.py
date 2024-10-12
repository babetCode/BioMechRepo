from adriensdir import BioMechDir
mydir = BioMechDir().getimufunctions()
from imufunctions import *
print(mydir)
my3dplot = plt.figure().add_subplot(projection='3d')

my3dplot.set_xlabel('x')

my3dplot.set_ylabel('y')

my3dplot.set_zlabel('z')

plot3axes(my3dplot)

plt.show()

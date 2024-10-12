import sys

sys.path.insert(0, 'c:\\Users\\goper\\Files\\vsCode\\490R\\BioMechRepo')

import matplotlib.pyplot as plt
from imufunctions import *

my3dplot = plt.figure().add_subplot(projection='3d')

my3dplot.set_xlabel('x')

my3dplot.set_ylabel('y')

my3dplot.set_zlabel('z')

plot3axes(my3dplot)

plt.show()
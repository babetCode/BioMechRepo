"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
Rotations in three-space                                               |
_______________________________________________________________________|
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
Author: Adrien Babet    GitHub: @babetcode    Email: adrienbabet1@gmail.com   |
______________________________________________________________________________|
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_quaternion(point, axis, angle: float) -> list:
    #print(np.linalg.norm(axis))
    normalizedaxis = axis/np.linalg.norm(axis)
    q_inv = [math.cos(angle/2)]
    q = [math.cos(angle/2)]
    for value in normalizedaxis:
        q.append(math.sin(angle/2)*value)
        q_inv.append(-math.sin(angle/2)*value)
    pointquat = [0, point[0], point[1], point[2]]
    rotation = quaternion_multiply(quaternion_multiply(q, pointquat), q_inv)
    result = [float(i) for i in rotation[1:]]
    return(result)

def plot_rotation(point, axis, angle, figure, name):
    t = np.linspace(0.0, angle, 100)
    x = np.zeros(100)
    y = np.zeros(100)
    z = np.zeros(100)
    for i in range(100):
        p = rotate_quaternion(point, axis, t[i])
        x[i] = p[0]
        y[i] = p[1]
        z[i] = p[2]
    figure.plot(x, y, z, label=name)
    return(rotate_quaternion(point, axis, angle))

def animate_rotation(point, axis, angle, figure, name):
    def update(frame):
        

    ani = FuncAnimation(figure, update,
        frames=np.linspace(0, angle, angle * 16), )


plt.close('all')
my3dplot = plt.figure().add_subplot(projection='3d')
my3dplot.set_xlabel('x')
my3dplot.set_ylabel('y')
my3dplot.set_zlabel('z')

def plot3axes(figure):
    figure.plot((-1.3,1.3), (0,0), (0,0), label='x')
    figure.plot((0,0), (-1.3,1.3), (0,0), label='y')
    figure.plot((0,0), (0,0), (-1.3,1.3), label='z')
point = np.array([1,0,0])
angle = 2.
axis = np.array([1,1,1])
plot_rotation(point, axis, angle, my3dplot, "test name")
plot3axes(my3dplot)
plot3axes(my3dplot)
plt.show()
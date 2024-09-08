"""
playground for learning quaternion rotation

quaternion multiplication rules:
i^2 = j^2 = k^2 = -1
ij = -ji = k
jk = -kj = i
ki = -ik = j

TO DO:
- remove print checks form rotateQuaternion()
- check for readability against chatGPT code
-normalize axis in rotateQuaternion()

"""

import numpy as np
from math import *

q1 = [0, 0, 1, 0]
q2 = [0, 0, 0, 1]

#multiply  two quaternions, given as 4-item lists
def multiplyQuaternion(q1, q2):
    [w1, x1, y1, z1] = [value for value in q1]
    [w2, x2, y2, z2] = [value for value in q2]
    scalar = w1*w2 - x1*x2 - y1*y2 - z1*z2
    i = w1*x2 + x1*w2 + y1*z2 - z1*y2
    j = w1*y2 + y1*w2 + z1*x2 - x1*z2
    k = w1*z2 + z1*w2 + x1*y2 -y1*x2
    return([scalar, i, j, k])

#given point as 3-item list, angle in radians, axix as 3-item list
def rotateQuaternion(point, angle, axis):
    p = [0]
    for value in point:
        p.append(value)
    print('p = ', p, '\n')
    q = [cos(angle/2)]
    q_inv = [cos(angle/2)]
    for value in axis:
        q.append(sin(angle/2)*value)
        q_inv.append(-sin(angle/2)*value)
    print('q = ', q, '\n')
    print('q_inverse = ', q_inv, '\n')
    rotation = multiplyQuaternion(multiplyQuaternion(q, p), q_inv)
    print('rotation = ', rotation, '\n')
    result = rotation[1:]
    print('result = ', result, '\n')
    return(result)


default_orientation = np.array([[1,0,0], [0,1,0], [0,0,1]])
default_position = np.array([0,0,0])
class imu:
    def __init__(self, name, initial_axes, initial_pos):
        self.name = name
        self.local_axes = initial_axes
        self.position = initial_pos
        self.display_axes = np.array(
        [[round(xyz, 2) for xyz in axis] for axis in self.local_axes])
        self.display_position = np.array([round(xyz, 2) for xyz in self.position])
    def __str__(self):
        return f"{self.name} \
        \norientation: x{self.display_axes[0]}, y{self.display_axes[1]}, z{self.display_axes[2]} \
        \nposition: {self.position}"
    def update_display(self):
        self.display_axes = np.array(
            [[round(xyz, 2) for xyz in axis] for axis in self.local_axes])
        self.display_position = np.array([round(xyz, 2) for xyz in self.position])
    def rotate(self, angle, axis):
        new_orientation = [rotateQuaternion(i, angle, axis) for i in self.local_axes]
        self.local_axes = new_orientation
        self.update_display()


        
def main():
    mydict = {}

#CALL MAIN
if __name__ == '__main__':
    main()
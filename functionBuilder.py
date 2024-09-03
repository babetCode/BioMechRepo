"this is a drafting space for function to be used in IMU gait analysis"

import numpy as np
from math import *

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
    normalizedAxis = axis/np.linalg.norm(axis)
    q_inv = [cos(angle/2)]
    q = [cos(angle/2)]
    q_inv = [cos(angle/2)]
    for value in normalizedAxis:
        q.append(sin(angle/2)*value)
        q_inv.append(-sin(angle/2)*value)
    pointQuat = [0, point[0], point[1], point[2]]
    rotation = multiplyQuaternion(multiplyQuaternion(q, pointQuat), q_inv)
    result = [float(i) for i in rotation[1:]]
    roundedPoint = [round(i, 3) for i in point]
    roundedAngle = round(angle, 3)
    roundedAxis = [round(i, 3) for i in axis]
    roundedResult = [round(i, 3) for i in result]
    #print('the rotation of ', roundedPoint, ' by ', roundedAngle, ' radians around ', roundedAxis, ' is ', roundedResult, '\n')
    return(result)

#make class for IMUs
default_orientation = np.array([[1,0,0], [0,1,0], [0,0,1]])
default_position = np.array([0,0,0])
class imu:
    def __init__(self, name, initial_orient, initial_pos):
        self.name = name
        self.orientation = initial_orient
        self.position = initial_pos
    def __str__(self):
        return f"{self.name}({self.orientation},{self.position})"
    def rotate(self, angle, axis):
        new_orientation = [rotateQuaternion(i, angle, axis) for i in self.orientation]
        self.orientation = new_orientation
def main():
    rotateQuaternion([23.000,0.000,0.000], pi, [0,sqrt(2)/2,sqrt(2)/2])

#CALL MAIN
if __name__ == '__main__':
    main()
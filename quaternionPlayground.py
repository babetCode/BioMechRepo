"""
playground for learning quaternion rotation

quaternion multiplication rules:
i^2 = j^2 = k^2 = -1
ij = -ji = k
jk = -kj = i
ki = -ik = j

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
    print('point is ', p, '\n')
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

def main():
    print(multiplyQuaternion([0,1,0,0], [0,1,0,0]))

#CALL MAIN
if __name__ == '__main__':
    main()
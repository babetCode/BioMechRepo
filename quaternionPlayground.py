"""
playground for learning quaternion rotation

quaternion multiplication rules:
i^2 = j^2 = k^2 = -1
ij = -ji = k
jk = -kj = i
ki = -ik = j

"""

import numpy as np
import math

q1 = [0, 0, 1, 0]
q2 = [0, 0, 0, 1]

#multiply  two quaternions, given as 4-item lists
def multiplyQuaternion(q1, q2):
    [w1, x1, y1, z1] = [value for value in q1]
    [w2, x2, y2, z2] = [value for value in q2]
    scalar = w1*w2 - x1*x2 - y1*y2 - z1*z2
    i = w1*x2 + x1*w2 + y1*z2 - z1*y2
    j = w1*y2 + y1*w2 + z1*x2 - x1*z2
    k = w1*z2 + z1*w1 + x1*y2 -y1*x2
    return([scalar, i, j, k])

def main():
    print(multiplyQuaternion(q1, q2))

#CALL MAIN
if __name__ == '__main__':
    main()
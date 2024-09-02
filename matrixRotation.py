import numpy as np
from math import *

#set angles
theta = pi/3
phi = 0

#make a 2d vector & rotation matrix
vector2d = np.array([[1],[0]])
rotation_matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

#rotate 2d vector counterclockwise by theta
def rotate2dvec(vec, theta):
    matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return(np.dot(matrix, vec))

#make a 3d vector
vec3d = np.array([[cos(theta)], [sin(theta)*cos(phi)], [sin(phi)]])

#rotate vector by yaw A, ptich B, roll C
def rotate3dvec(vec, A, B, C):
    row1 = [cos(A)*cos(B), cos(A)*sin(B)*sin(C)-sin(A)*cos(C), cos(A)*sin(B)*cos(C)+sin(A)*sin(B)]
    row2 = [sin(A)*cos(B), sin(A)*sin(B)*sin(C)+cos(A)*cos(C), sin(A)*sin(B)*cos(C)-cos(A)*sin(C)]
    row3 = [-sin(B), cos(B)*sin(C), cos(B)*cos(C)]
    matrix = np.array([row1, row2, row3])
    return(np.dot(matrix, vec))

#rotate around x, y, z, axes
def rotateAxes(vector, x, y, z):
    vec = vector
    print(vec, '\n \n')
    r_x = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    r_y = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    r_z = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    for i in [r_x, r_y, r_z]:
        vec = np.dot(i, vec)
        print(vec, '\n \n')


#convert spherical coordinates to cartesian coordinates
def sphere2cart():
    print('empty')

"""
DEFINE MAIN FUNCTION
"""
def main():

    rotateAxes(vec3d, pi, pi, pi/6)    

#call main
if __name__ == "__main__":
    main()
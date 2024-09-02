import numpy as np
from math import *

vector2d = np.array([[1],[0]])

theta = pi/3

rotation_matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

#rotate 2d vector counterclockwise by theta
def rotate2dvec(vec, theta):
    matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return(np.dot(matrix, vec))

#rotate vector by yaw A, ptich B, roll C
def rotate3dvec(vec, A, B, C):
    row1 = [cos(A)*cos(B), cos(A)*sin(B)*sin(C)-sin(A)*cos(C), cos(A)*sin(B)*cos(C)+sin(A)*sin(B)]
    row2 = [sin(A)*cos(B), sin(A)*sin(B)*sin(C)+cos(A)*cos(C), sin(A)*sin(B)*cos(C)-cos(A)*sin(C)]
    row3 = [-sin(B), cos(B)*sin(C), cos(B)*cos(C)]
    matrix = np.array([row1, row2, row3])
    return(np.dot(matrix, vec))

#rotate around x, y, z, axes
def rotateAxes(vec, x, y, z):
    r_x = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    r_y = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    r_z = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    print(r_x, '\n \n', r_y, '\n \n', r_z)

#convert spherical coordinates to cartesian coordinates


#create main function
def main():

    rotateAxes(1, 1, 1, 1)    

#call main
if __name__ == "__main__":
    main()
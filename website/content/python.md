---
date: '2025-01-28T18:38:45-07:00'
draft: false
title: 'Using Python'
type: docs
breadcrumbs: false
weight: 2
math: true
---

Python is ideal for building a Kalman filter due to its simplicity, readability, and robust libraries for linear algebra and data analysis.

## Getting started with NumPy
NumPy arrays are n-dimensional data structures that enable efficient computations and are particularly well-suited for handling the matrix manipulations required in Kalman filtering.

### Matrix Multiplication
Consider the expression
$$\begin{bmatrix}6&2&4 \\\ -1&4&3 \\\ -2&9&3\end{bmatrix}
\begin{bmatrix}4 \\\ -2 \\\ 1\end{bmatrix}.$$

We can evaluate this with NumPy using:
```py
import numpy as np

# Define the matrix
matrix = np.array([[6, 2, 4],
                   [-1, 4, 3],
                   [-2, 9, 3]])

# Define the vector
vector = np.array([[4], [-2], [1]])

# Perform the matrix-vector multiplication
result = matrix @ vector  # Alternatively, use np.dot(matrix, vector)

print(result)
```
This will output:  
<span style="font-family:monospace">[[ 24]  
&nbsp;[ -9]  
&nbsp;[-23]]</span>

### Measurement & State Vectors
Consider our vectors
$$
\mathbf x_k = \begin{bmatrix} p^{\text{N}}_k \\\ p^{\text{E}}_k \\\ p^{\text{D}}_k \\\ v^{\text{N}}_k \\\ v^{\text{E}}_k \\\ v^{\text{D}}_k \\\ a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k \\\ q^0_k \\\ q^1_k \\\ q^2_k \\\ q^3_k \\\ \omega^{\text{N}}_k \\\ \omega^{\text{E}}_k \\\ \omega^{\text{D}}_k \end{bmatrix},\ \ 
\mathbf z_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \\\ \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix}.
$$
We can represent these with:
```py
# initial state
x_0 = np.array([[0], [0], [0],
                [0], [0], [0],
                [0], [0], [0],
                [1], [0], [0], [0],
                [0], [0], [0]])

# initial measurements
z_0 = np.array([[0], [0], [0],
                [0], [0], [0]])
```

### Rotations
We can find our rotation matrix $\mathbf C$ using:
```py
def c_matrix(quaternion: np.ndarray) -> np.ndarray:
    if len(quaternion) != 4:
        raise ValueError(f"Expected quaternion of length 4, \
                         got {len(quaternion)} instead")

    quaternion = quaternion.reshape(-1, 1)
    q0 = quaternion[0, 0]
    q1 = quaternion[1, 0]
    q2 = quaternion[2, 0]
    q3 = quaternion[3, 0]

    c = np.array([[1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                  [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]])

    return c
```
As a basic check, we can verify the identity quaternion from <span style="font-family:monospace">x_0</span> produces the identify matrix:
```py
quat = x_0[9:13]
print(c_matrix(quat))
```
This should return:  
<span style="font-family:monospace">[[1 0 0]  
&nbsp;[0 1 0]  
&nbsp;[0 0 1]]</span>

More generally, a quaternion of the form $\left[\cos(\theta/2) + \sin(\theta/2)\left(xi + yj + zk\right)\right]$ represents a rotation of $\theta$ around the vector $\langle x,y,z \rangle$.

<!-- One of the easiest ways to implement kamlan filters in python is using Roger Labbe's [FilterPy](https://filterpy.readthedocs.io/en/latest/) library. -->
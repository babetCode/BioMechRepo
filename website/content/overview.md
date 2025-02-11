---
date: '2025-01-28T18:24:22-07:00'
draft: false
title: 'Mathematical Overview'
math: true
type: docs
breadcrumbs: false
weight: 1
---

## Measurement Variable

When using an IMU for gait analysis, we would like to use the IMU's measurements to calculate heel-strike, toe-off, and stride length (and perhaps we'll add toe-down and heel-off if we're feeling ambitious). At any given time $k$, the IMU will give us accelerometer data along its three local axes. We can think of this accereration data as a vector $\mathbf a^\text{local}$, where at time $k$, we have
$$
\mathbf a^{\text{local}}_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \end{bmatrix}.
$$

It will also give us rotational velocity along these local axes which we can write as
$$
\boldsymbol\omega^{local}_k = \begin{bmatrix} \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix}.
$$

Putting these together, we can think of our measurements as being represented by a variable $\mathbf z$, where at time $k$ the IMU gives us the reading
$$
\mathbf z_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \\\ \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix}.
$$

It's important to keep in mind that these measurements are with respect to the local frame of the IMU, and not the world frame.

## State Variable

In order to determine when and how gait events happen, we would need to know the IMU's position and orientation in world frame axes, such as north($N$)-east($E$)-down($D$) axes. Additionally, it would be nice to have the IMU's velocity and acceleration in the world frame. To visualize this, we could assign variables to position, linear velocity, linear acceleration, orientation, and angular velocity, like this:
$$
\begin{align*}
\mathbf p^{\text{world}}_k &= \begin{bmatrix} p^{\text{N}}_k \\\ p^{\text{E}}_k \\\ p^{\text{D}}_k \end{bmatrix}, \\\\
\mathbf v^{\text{world}}_k &= \begin{bmatrix} v^{\text{N}}_k \\\ v^{\text{E}}_k \\\ v^{\text{D}}_k \end{bmatrix}, \\\\
\mathbf a^{\text{world}}_k &= \begin{bmatrix} a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k \end{bmatrix}, \\\\
\mathbf q^{\text{world}}_k &= \begin{bmatrix} q^0_k \\\ q^1_k \\\ q^2_k \\\ q^3_k \end{bmatrix}, \\\\
\boldsymbol\omega^{\text{world}}_k &= \begin{bmatrix} \omega^{\text{N}}_k \\\ \omega^{\text{E}}_k \\\ \omega^{\text{D}}_k \end{bmatrix}.
\end{align*}
$$

Here, $\mathbf q_k^\text{world}$ is a vector representation of the quaternion $\left[q^0_k + i\left(q^1_k\right) + j\left(q^2_k\right) + k\left(q^3_k\right)\right]$. We use quaternions rather than matricies to represent orientation because they let us update our orientation using the quaternion update function
$$
\mathbf q_{k+1} = \mathbf q_k+\frac12dt\cdot\mathbf q_k\otimes\left[0 + i\left(\omega^{\text{N}}_k\right) + j\left(\omega^{\text{E}}_k\right) + k\left(\omega^{\text{D}}_k\right)\right],
$$
providing a consistent method for interpolating angles between time steps *($\otimes$ represents the "[Hamilton product](https://en.wikipedia.org/wiki/Quaternion#:~:text=of%20vector%20quaternions.-,Hamilton%20product,-%5Bedit%5D)", AKA quaternion multiplication).*

Putting these together, we can think of our system state (at least the parts we care about) as being represented by a variable $\mathbf x$, where at time $k$ we estimate that its properties are
$$
\mathbf x_k = \begin{bmatrix} p^{\text{N}}_k \\\ p^{\text{E}}_k \\\ p^{\text{D}}_k \\\ v^{\text{N}}_k \\\ v^{\text{E}}_k \\\ v^{\text{D}}_k \\\ a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k \\\ q^0_k \\\ q^1_k \\\ q^2_k \\\ q^3_k \\\ \omega^{\text{N}}_k \\\ \omega^{\text{E}}_k \\\ \omega^{\text{D}}_k \end{bmatrix}.
$$

### Notes on Quaternions:
{{< details-html title="What are Quaternions?" closed="true" >}}
{{< md >}}


Quaternions are four-dimensional hypercomplex numbers which offer a powerful approach to orientation tracking because of their capacity for representing three-dimensional rotations. Specifically, a single unit quaternion can represent any 3d rotation without the common pitfalls of other rotation representations, such as gimbal lock. Moreover, quaternions allow for efficient and numerically stable calculations, especially beneficial in real-time tracking contexts like gait analysis. The quaternion-based approach simplifies the compounding of rotations by encapsulating complex rotational transformations in quaternion multiplication, reducing the risk of drift and improving accuracy in sensor fusion algorithms. This capability makes quaternions particularly well-suited for the recursive nature of state estimation models, where orientation data from consecutive IMU measurements need to be seamlessly integrated over time.

Quaternions also support interpolation methods such as spherical linear interpolation (SLERP), which preserves the shortest path of rotation and minimizes error, critical in applications like gait analysis where precise orientation tracking is needed. This combination of stability, efficiency, and the ability to handle continuous rotations makes quaternions an optimal choice for robust and reliable orientation tracking in IMU-based gait analysis.

Quaternions represent orientation through a four-dimensional structure that encodes three-dimensional rotation in a single unit quaternion, commonly denoted $q = w + xi + yj + zk$, where $w$, $x$, $y$, and $z$ are real numbers and $i$, $j$, and $k$, are imaginary units. The product $ijk$ is defined to equal $-1$, and multiplication between any two imaginaries follows the rules of the table below, where the left column shows the left factor and the top row shows the right factor (note that the products are *not* commutative):
| $\boldsymbol\otimes$ | $\textbf{\textit i}$ | $\textbf{\textit j}$ | $\textbf{\textit k}$ |
|----------------------|----------------------|----------------------|----------------------|
| $\textbf{\textit i}$ |  $-1$   |   $k$   |  $-j$   |
| $\textbf{\textit j}$ |  $-k$   |  $-1$   |   $i$   |
| $\textbf{\textit k}$ |   $j$   |  $-i$   |  $-1$   |

Multiplication between two quaternions $q_1 = w_1 + x_1i + y_1j + z_1k$ and $q_2 = w_2 + x_2i + y_2j + z_2k$, is defined according to the Hamilton product formula:
$$\begin{align*}
q=q_1 \otimes q_2=&\ \ \left(w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2\right) \\\
&+ \left(w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2\right)i \\\
&+ \left(w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2\right)j \\\
&+ \left(w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2\right)k,
\end{align*}$$
which is also *not* commutative.

Quaternion multiplication is useful in representing 3d rotations because it can easily compute a rotation by a given angle around some axis passing through the origin. Consider the rotation of a point $P$ by an angle of $\theta$ around an axis $A$ which passes through the origin. First, we normalize the vector $\overrightarrow A$ such that if it has components $x$, $y$, $z$, then $x^2+y^2+z^2=1$. Next, we construct a quaternion $q$ such that
$$q = (\cos(\theta/2)+\sin(\theta/2)(xi+yj+zk)),$$
and its inverse $q^{-1}$ such that
$$q^{-1} = (\cos(\theta/2)-\sin(\theta/2)(xi+yj+zk)).$$
Now, we make a quaternion for our point $P$ so that if our point $P$ had coordinates $a$, $b$, $c$, then our quaternion $p$ is defined by
$$p = ai+bj+ck.$$
Now, to find the projection $P^\prime$ after the rotation, we have
$$P^\prime = q\otimes p\otimes q^{-1}.$$
By convention, quaternions are used to represent orientation by representing the rotation necessary for an object to go from some predefined "neutral" to its current orientation. In other words, a quaternion of the form $\left[\cos(\theta/2) + \sin(\theta/2)\left(xi + yj + zk\right)\right]$ represents a rotation from neutral of $\theta$ around the vector $\langle x,y,z \rangle$.

This framework for how quaternions work is admittedly not intuitive. The videos below from 3Blue1Brown provide a more in depth explanation of the mechanics, with helpful visuals.

Part 1:
{{< /md >}}
{{< youtube d4EgbgTm0Bg >}}
{{< md >}}Part 2:{{< /md >}}
{{< youtube zjMuIxRvygQ >}}
{{< /details-html >}}

## Translating Between Local and World Frames

Our goal is to use our local pitch-roll-yaw coordinate system measurements to estimate the system state in terms of the global coordinate system. A difficulty with calculating acceleration in this manner is that the direction of gravity will change as our local axes rotate, and our accelerometers will not be able to distinguish this change in gravity from a change in linear acceleration. In this section, we use our quaternion orientation to address the issue.

At a given time $k$, we will have the IMU's orientation stored as a quaterion $\mathbf q_k$ which represents the rotation from a "neutral" orientation to IMU's current orientation. In order to calculate the "down" direction from this, it is most efficient to convert this quaternion to a matrix. The rotation matrix $\mathbf C_k$, defined as
$$
\mathbf C_k = \begin{bmatrix}
1 - 2\big((q^2_k)^2 + (q^3_k)^2\big) & 2\big(q^1_k q^2_k - q^0_k q^3_k\big) & 2\big(q^1_k q^3_k + q^0_k q^2_k\big) \\\\
2\big(q^1_k q^2_k + q^0_k q^3_k\big) & 1 - 2\big((q^1_k)^2 + (q^3_k)^2\big) & 2\big(q^2_k q^3_k - q^0_k q^1_k\big) \\\\
2\big(q^1_k q^3_k - q^0_k q^2_k\big) & 2\big(q^2_k q^3_k + q^0_k q^1_k\big) & 1 - 2\big((q^1_k)^2 + (q^2_k)^2\big)
\end{bmatrix},
$$
rotates a vector from the local frame to the world frame. In other words, we have
$$
\begin{align*}
\mathbf a^{\text{world}}_k = \mathbf C_k \cdot \mathbf a^{\text{local}}_k, \\\\
\boldsymbol\omega^{\text{world}}_k = \mathbf C_k \cdot \boldsymbol\omega^{\text{local}}_k.
\end{align*}
$$
Furthermore, since $\mathbf C_k$ is an orthogonal matrix, its inverse must be equal to its transpose $\mathbf C^T_k$, and thus
$$
\begin{align*}
\mathbf a^{\text{local}}_k = \mathbf C^T_k \cdot \mathbf a^{\text{world}}_k, \\\\
\boldsymbol\omega^{\text{local}}_k = \mathbf C^T_k \cdot \boldsymbol\omega^{\text{world}}_k.
\end{align*}
$$
Because of this, we can calculate world frame acceleration from our local measurements in a way that accounts for gravity. If we are measuring acceleration in m/s^2 units then we will always have
$$\mathbf a^{\text{world}}_k = \begin{bmatrix} a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k \end{bmatrix} = \begin{bmatrix} 0 \\\ 0 \\\ 9.81 \end{bmatrix}.$$

Therefore, a stationary sensor with any given pitch-roll-yaw axes should read
$$
\mathbf a^{\text{local}}_k = \mathbf C^T_k \begin{bmatrix} 0 \\\ 0 \\\ 9.81 \end{bmatrix}.
$$
By extension, any deviation from this value means that the sensor is accelerating in the world frame, so at any time $k$ our sensor should read
$$
\mathbf a^{\text{local}}_k = \mathbf C^T_k \left(\mathbf a^{\text{world}}_k + \begin{bmatrix}0 \\ 0 \\ 9.8\end{bmatrix}\right).
$$
This looks like exactly what we need! To make things more concise, we will add $\mathbf a^{\text{world}}$ and gravity together into one vector, and write
$$
\mathbf a^{\text{local}}_k = \mathbf C^T_k \begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9.8\end{bmatrix}.
$$

Now that we've defined our problem and seen a bit of how our measurment and state variables relate to each other, it's time to build a sensor fusion algorith to estimate state from measurements.

## Kalman Filter

The kalman filter estimates an nx1 column vector state variable ($\mathbf x$), based on some mx1 column vector measurement ($\mathbf z$), using a system model:
$$
\begin{align*}
\text{state transition matrix} &: \mathbf A  &(& \text{nxn matrix}), \\\\
\text{process noise covariance} &: \mathbf Q  &(& \text{nxn diagonal matrix}), \\\\
\text{measurement covariance} &: \mathbf C  &(& \text{mxm matrix}), \\\\
\text{measurement model matrix} &: \mathbf H  &(& \text{mxn matrix}).
\end{align*}
$$

After the system model has been set, there are five steps of the simple kalman filter:

0. <u>Set initial values</u>
$$
\begin{align*}
\mathbf x_0 &= \text{initial state} &(& \text{nx1 column vector}), \\\\
\mathbf P_0 &= \text{initial error covariance} &(& \text{nxn matrix}).
\end{align*}
$$

1. <u>Predict state and error covariance:</u>
$$
\begin{align*}
\mathbf{\bar x_k} &= \mathbf A \mathbf x_{k-1}, \\\\
\mathbf{\bar P_k} &= \mathbf A \mathbf P_{k-1} \mathbf A^T + \mathbf Q.
\end{align*}
$$

2. <u>Compute kalman gain:</u>
$$
\mathbf K_k = \mathbf{\bar P}_k \mathbf H^T \left(\mathbf H \mathbf{\bar P}_k \mathbf H^T + \mathbf R\right)^{-1}.
$$

3. <u>Compute the estimate (state update equation):</u>
$$
\mathbf x_k = \mathbf{\bar x}_k + \mathbf K_k \left(\mathbf z_k - \mathbf H \mathbf{\bar x}_k\right).
$$

4. <u>Compute the error covariance:</u>
$$
\mathbf P_k = \mathbf{\bar P}_k - \mathbf K_k \mathbf H \mathbf{\bar P}_k.
$$

Steps 1-4 are then repeated to recursively update with each new $\mathbf z_k$.

*Notes:*
1. *Here, the bar notations denote predicted values before measurement.*

2. *The term $(\mathbf z_k - \mathbf H \mathbf{\bar x}_k)$ in the state update equation is important, because it represents the gap between our prediction, and our measurement. Because of this importance, it is given the name "measurement residual" or "innovation".*

This can be applied to both uni-variate and multi-variate systems, and the notation is unfortunates not always consistent. Below is an explanation from Roger Labbe in his book [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) (note that Labbe refers to the state transition as $\mathbf F$ rather than $\mathbf A$):

>"[the univariate and multivariate equations]... are quite similar.
>
><u>**Predict**</u>
>
>
>$
\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\\\
& \text{(Kalman form)} & \\\\
\hline
\bar \mu = \mu + \mu_{f_x} & \bar x = x + dx & \bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}\\\\
\bar\sigma^2 = \sigma_x^2 + \sigma_{f_x}^2 & \bar P = P + Q & \bar{\mathbf P} = \mathbf{FPF}^\mathsf T + \mathbf Q >\\\\
\hline
\end{array}
$
>
>Without worrying about the specifics of the linear algebra, we can see that:
$\mathbf x,\, \mathbf P$ are the state mean and covariance. They correspond to $x$ and $\sigma^2$.
$\mathbf F$ is the *state transition function*. When multiplied by $\bf x$ it computes the prior.
$\mathbf Q$ is the process covariance. It corresponds to $\sigma^2_{f_x}$.
$\mathbf B$ and $\mathbf u$ are new to us. They let us model control inputs to the system.
>
><u>**Update**</u>
>
>
>$
>\begin{array}{|l|l|l|}
>\hline
>\text{Univariate} & \text{Univariate} & \text{Multivariate}\\\\
>& \text{(Kalman form)} & \\\\
>\hline
>& y = z - \bar x & \mathbf y = \mathbf z - \mathbf{H\bar x} \\\\
>& K = \frac{\bar P}{\bar P+R}&
>\mathbf K = \mathbf{\bar{P}H}^\mathsf T (\mathbf{H\bar{P}H}^\mathsf T + \mathbf R)^{-1} \\\\
>\mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2} & x = \bar x + Ky & \mathbf x = \bar{\mathbf x} + \mathbf{Ky} \\\\
>\sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} & P = (1-K)\bar P & \mathbf P = (\mathbf I -\mathbf{KH})\mathbf{\bar{P}} \\\\
>\hline
>\end{array}
>$
>
>$\mathbf H$ is the measurement function. We haven't seen this yet in this book and I'll explain it later. If you mentally remove $\mathbf H$ from the equations, you should be able to see these equations are similar as well.
>
>$\mathbf z, \mathbf R$ are the measurement mean and noise covariance. They correspond to $z$ and $\sigma_z^2$ in the univariate filter (I've substituted $\mu$ with $x$ for the univariate equations to make the notation as similar as possible).
>
>$\mathbf y$ and $\mathbf K$ are the residual and Kalman gain.
>
>The details will be different than the univariate filter because these are vectors and matrices, but the concepts are exactly the same:
>-  Use a Gaussian to represent our estimate of the state and error
>-  Use a Gaussian to represent the measurement and its error
>-  Use a Gaussian to represent the process model
>-  Use the process model to predict the next state (the prior)
>-  Form an estimate part way between the measurement and the prior
>Your job as a designer will be to design the state $\left(\mathbf x, \mathbf P\right)$, the process $\left(\mathbf F, \mathbf Q\right)$, the measurement $\left(\mathbf z, \mathbf R\right)$, and the measurement function $\mathbf H$. If the system has control inputs, such as a robot, you will also design $\mathbf B$ and $\mathbf u$."

Lets try applying this to our problem.

## State

The state of the kalman filter is described by state variable $\mathbf x$ and the covariance $\mathbf P$. In this section, we will discuss how to set their initial values. After we set their initial values, our kalman filter will update them internally at each time step.

### x

As described in the *"State Variable"* section, we want $\mathbf x$ to be a be a 16x1 vector. If we could set our origin at the initial position of the IMU, and we could be fairly certain that it would be stationary and aligned with $N$-$E$-$D$ axes when we start recording data, then a resonable initial state $\mathbf x_0$ might look like:
$$
\mathbf p^{\text{world}}_k = \begin{bmatrix}0 \\\ 0 \\\ 0\end{bmatrix},\ \ 
\mathbf v^{\text{world}}_k = \begin{bmatrix}0 \\\ 0 \\\ 0\end{bmatrix},\ \ 
\mathbf a^{\text{world}}_k = \begin{bmatrix}0 \\\ 0 \\\ 0\end{bmatrix},\ \ 
\mathbf q^{\text{world}}_k = \begin{bmatrix}1 \\\ 0 \\\ 0 \\\ 0\end{bmatrix},\ \ 
\boldsymbol\omega^{\text{world}}_k = \begin{bmatrix}0 \\\ 0 \\\ 0\end{bmatrix},
$$
$$
\implies\mathbf x_0 = \begin{bmatrix}0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 1 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0 \\\ 0\end{bmatrix}.
$$

### P

The state covariance $\mathbf P$ will be a 16x16 (or 13x13) matrix which represents the covariance of the state. A reasonable $\mathbf P_0$ would be:
$$
\begin{bmatrix}
\sigma_{p_0^{\text{N}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & \sigma_{p_0^{\text{E}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & \sigma_{p_0^{\text{D}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & \sigma_{v_0^{\text{N}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & \sigma_{v_0^{\text{E}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & \sigma_{v_0^{\text{D}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & \sigma_{a_0^{\text{N}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{a_0^{\text{E}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{a_0^{\text{D}}}^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{q_0^0}^2 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{q_0^1}^2 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{q_0^2}^2 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{q_0^3}^2 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{\omega_0^{\text{N}}}^2 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{\omega_0^{\text{E}}}^2 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma_{\omega_0^{\text{D}}}^2 \\\\
\end{bmatrix},
$$
where $\sigma^2_{p^N_0}$ is the variance in the initial position in the north direction, and so on and so forth. As a general rule of thumb, it will be better to overestimate than underestimate --- the filter will converge if $\mathbf P_0$ is too large, but might not if it's too small.

## Process

The process of the kalman filter is described by $\mathbf F$ (the state transition function) and $\mathbf Q$ (the process covariance).

### F

Since we do not have any predetermined control inputs here, we want a 16x16 matrix $\mathbf F$ which we can multiply by the current state to get our predicted state in the next time step. To visualize this, we want a matrix which satisfies
$$
\begin{bmatrix}?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\ \end{bmatrix}
\begin{bmatrix}
p^\text N_{k} \\\
p_k^\text E \\\
p_k^\text D \\\
v_k^\text N \\\
v_k^\text E \\\
v_k^\text D \\\
a_k^\text N \\\
a_k^\text E \\\
a_k^\text D \\\
q_k^0 \\\
q_k^1 \\\
q_k^2 \\\
q_k^3 \\\
\omega_k^\text{N} \\\
\omega_k^\text{E} \\\
\omega_k^\text{D} \\\
\end{bmatrix}
= \begin{bmatrix}
p^\text N_{k+1} \\\
p^\text E_{k+1} \\\
p^\text D_{k+1} \\\
p^\text N_{k} \\\
v_{k+1}^\text E \\\
v_{k+1}^\text D \\\
a_{k+1}^\text N \\\
a_{k+1}^\text E \\\
a_{k+1}^\text D \\\
q_{k+1}^0 \\\
q_{k+1}^1 \\\
q_{k+1}^2 \\\
q_{k+1}^3 \\\
\omega_{k+1}^\text{N} \\\
\omega_{k+1}^\text{E} \\\
\omega_{k+1}^\text{D} \\\
\end{bmatrix}.
$$

There are a few things we need to make this happen.

#### 1. Position Update

Since the time between measurements is $dt$, then we want it to update the postion in a way that satisfies
$$
\mathbf p_{k+1} = \mathbf p_k + (\mathbf v_k)dt,
$$ 
where $dt$ is the time step between measurements. This expands to
$$
\begin{bmatrix}p^\text N_{k+1} \\\ p^\text E_{k+1} \\\ p^\text D_{k+1}\end{bmatrix} = \begin{bmatrix}p^\text N_{k} \\\ p^\text E_{k} \\\ p^\text D_{k}\end{bmatrix} + \begin{bmatrix}v^\text N_{k} \\\ v^\text E_{k} \\\ v^\text D_{k}\end{bmatrix}dt.
$$
Therefore, the top three rows of our matrix will be
$$
\begin{bmatrix}
1&0&0&dt&0&0&0&0&0&0&0&0&0&0&0&0\\\\
0&1&0&0&dt&0&0&0&0&0&0&0&0&0&0&0\\\\
0&0&1&0&0&dt&0&0&0&0&0&0&0&0&0&0
\end{bmatrix}.
$$

#### 2. Velocity Update

We also want it to update the velocity in way that satisfies
$$
\mathbf v_{k+1} = \mathbf v_k + (\mathbf a_k)dt,
$$
which similarly expands to
$$
\begin{bmatrix}v_{k+1}^\text N \\\ v_{k+1}^\text E \\\ v_{k+1}^\text D\end{bmatrix} = \begin{bmatrix}v_{k}^\text N \\\ v_{k}^\text E \\\ v_{k}^\text D\end{bmatrix} + \begin{bmatrix}a_{k}^\text N \\\ a_{k}^\text E \\\ a_{k}^\text D\end{bmatrix} dt,
$$
and similarly gives us the next three rows of our matrix
$$
\begin{bmatrix}
0&0&0&1&0&0&dt&0&0&0&0&0&0&0&0&0\\\\
0&0&0&0&1&0&0&dt&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&1&0&0&dt&0&0&0&0&0&0&0
\end{bmatrix}.
$$

#### 3. Acceleration Update

To keep things simple, we won't predict change here, so we'll use
$$
\mathbf a_{k+1} = \mathbf a_k.
$$
Therefore, next three rows of our matrix will be
$$
\begin{bmatrix}
0&0&0&0&0&0&1&0&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&1&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&0&1&0&0&0&0&0&0&0
\end{bmatrix}.
$$

#### 4. Orientation Update

We want it to update the orientation in a way that satisfies the quaternion update function
$$
\mathbf q_{k+1} = \mathbf q_k+\frac12dt\cdot\mathbf q_k\otimes\begin{bmatrix}0 \\ \omega x_k \\ \omega y_k \\ \omega z_k\end{bmatrix}.
$$
satisfy our rotation update equation.Let's start by expanding our quaternion multiplication term.

We know that the product of two quaternions
$$
\mathbf q_1 = (w_1 + x_1i + y_1j + z_1k)
$$
and
$$
\mathbf q_2 = (w_2 + x_2i + y_2j + z_2k)
$$
is calculated using the formula:
$$
\begin{align*}
\mathbf q=\mathbf q_1 \otimes \mathbf q_2=&\ \ \ \left(w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2\right) \\\\
&+ \left(w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2\right)i \\\\
&+ \left(w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2\right)j \\\\
&+ \left(w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2\right)k.
\end{align*}
$$
Substituting
$$
\begin{align*}
\begin{bmatrix}w_1\\ x_1\\ y_1\\ z_1\end{bmatrix} &= \begin{bmatrix}q^0_k\\ q^1_k\\ q^2_k\\ q^3_k\end{bmatrix},\\\\
\begin{bmatrix}w_2\\ x_2\\ y_2\\ z_2\end{bmatrix} &= \begin{bmatrix}0 \\ \omega^N_k \\ \omega^E_k \\ \omega^D_k\end{bmatrix},
\end{align*}
$$
gives
$$
\begin{align*}
\mathbf q_k\otimes\begin{bmatrix}0 \\ \omega^N_k \\ \omega^E_k \\ \omega^D_k\end{bmatrix} &= \begin{bmatrix}q^0_k\\ q^1_k\\ q^2_k\\ q^3_k\end{bmatrix}\begin{bmatrix}0 \\ \omega^N_k \\ \omega^E_k \\ \omega^D_k\end{bmatrix} \\\\
&=\ \ \ \left(q^0_k0 - q^1_k\omega^N_k - q^2_k\omega^E_k - q^3_k\omega^D_k\right)  \\\\
&\ \ \ \ + \left(q^0_k\omega^N_k + q^1_k0 + q^2_k\omega^D_k - q^3_k\omega^E_k\right)i  \\\\
&\ \ \ \ + \left(q^0_k\omega^E_k - q^1_k\omega^D_k + q^2_k0 + q^3_k\omega^N_k\right)j  \\\\
&\ \ \ \ + \left(q^0_k\omega^D_k + q^1_k\omega^E_k - q^2_k\omega^N_k + q^3_k0\right)k.
\end{align*}
$$
Writing this result in vector form, we have
$$
\begin{bmatrix}
(q^0_k)(0) &- (q^1_k)(\omega^N_k) &- (q^2_k)(\omega^E_k) &- (q^3_k)(\omega^D_k)\\\\
(q^0_k)(\omega^N_k) &+ (q^1_k)(0) &+ (q^2_k)(\omega^D_k) &- (q^3_k)(\omega^E_k)\\\\
(q^0_k)(\omega^E_k) &- (q^1_k)(\omega^D_k) &+ (q^2_k)(0) &+ (q^3_k)(\omega^N_k)\\\\
(q^0_k)(\omega^D_k) &+ (q^1_k)(\omega^E_k) &- (q^2_k)(\omega^N_k) &+ (q^3_k)(0)
\end{bmatrix}.
$$
We see that each component is in the form $[a(q0_k)+b(q1_k)+c(q2_k)+d(q3_k)]$, for some constants $a$, $b$, $c$, and $d$. This is looking quite close to the form we would like for our state transition matrix! We can substite
$$
\mathbf q_k\otimes\begin{bmatrix}0 \\ \omega_x \\ \omega_y \\ \omega_z\end{bmatrix} = \begin{bmatrix}
(q^0_k)(0) &- (q^1_k)(\omega^N_k) &- (q^2_k)(\omega^E_k) &- (q^3_k)(\omega^D_k)\\\\
(q^0_k)(\omega^N_k) &+ (q^1_k)(0) &+ (q^2_k)(\omega^D_k) &- (q^3_k)(\omega^E_k)\\\\
(q^0_k)(\omega^E_k) &- (q^1_k)(\omega^D_k) &+ (q^2_k)(0) &+ (q^3_k)(\omega^N_k)\\\\
(q^0_k)(\omega^D_k) &+ (q^1_k)(\omega^E_k) &- (q^2_k)(\omega^N_k) &+ (q^3_k)(0)
\end{bmatrix}
$$
into our rotation update equation to get
$$
\mathbf q_{k+1} = \mathbf q_k+\frac12dt\cdot
\begin{bmatrix}
(q^0_k)(0) &- (q^1_k)(\omega^N_k) &- (q^2_k)(\omega^E_k) &- (q^3_k)(\omega^D_k)\\\\
(q^0_k)(\omega^N_k) &+ (q^1_k)(0) &+ (q^2_k)(\omega^D_k) &- (q^3_k)(\omega^E_k)\\\\
(q^0_k)(\omega^E_k) &- (q^1_k)(\omega^D_k) &+ (q^2_k)(0) &+ (q^3_k)(\omega^N_k)\\\\
(q^0_k)(\omega^D_k) &+ (q^1_k)(\omega^E_k) &- (q^2_k)(\omega^N_k) &+ (q^3_k)(0)
\end{bmatrix}.
$$

Writing the whole right side as one vector gives
$$
\mathbf q_{k+1} = \begin{bmatrix}
q^0_k + (dt/2)((q^0_k)(0) - (q^1_k)(\omega^N_k) - (q^2_k)(\omega^E_k) - (q^3_k)(\omega^D_k)) \\\\
q^1_k + (dt/2)((q^0_k)(\omega^N_k) + (q^1_k)(0) + (q^2_k)(\omega^D_k) - (q^3_k)(\omega^E_k)) \\\\
q^2_k + (dt/2)((q^0_k)(\omega^E_k) - (q^1_k)(\omega^D_k) + (q^2_k)(0) + (q^3_k)(\omega^N_k)) \\\\
q^3_k + (dt/2)((q^0_k)(\omega^D_k) + (q^1_k)(\omega^E_k) - (q^2_k)(\omega^N_k) + (q^3_k)(0))
\end{bmatrix}.
$$
Therefore, next four rows of our matrix will be
$$
\begin{bmatrix}
0&0&0&0&0&0&0&0&0&1&-(dt\cdot\omega^N_k)/2&-(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^D_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^N_k)/2&1&(dt\cdot\omega^D_k)/2&-(dt\cdot\omega^E_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^D_k)/2&1&(dt\cdot\omega^N_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^D_k)/2&(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^N_k)/2&1&0&0&0
\end{bmatrix}.
$$

#### 5. Angular Velocity Update

We'll keep things simple here too by letting
$$
\boldsymbol\omega_{k+1} = \boldsymbol\omega_k.
$$
Therefore, last three rows of our matrix will be
$$
\begin{bmatrix}
0&0&0&0&0&0&0&0&0&0&0&0&0&1&0&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&1&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&0&1
\end{bmatrix}.
$$

We now have the matrix $\mathbf F$:
$$
\begin{bmatrix}
1&0&0&dt&0&0&0&0&0&0&0&0&0&0&0&0\\\\
0&1&0&0&dt&0&0&0&0&0&0&0&0&0&0&0\\\\
0&0&1&0&0&dt&0&0&0&0&0&0&0&0&0&0\\\\
0&0&0&1&0&0&dt&0&0&0&0&0&0&0&0&0\\\\
0&0&0&0&1&0&0&dt&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&1&0&0&dt&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&1&0&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&1&0&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&0&1&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&0&0&1&-(dt\cdot\omega^N_k)/2&-(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^D_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^N_k)/2&1&(dt\cdot\omega^D_k)/2&-(dt\cdot\omega^E_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^D_k)/2&1&(dt\cdot\omega^N_k)/2&0&0&0\\\\
0&0&0&0&0&0&0&0&0&(dt\cdot\omega^D_k)/2&(dt\cdot\omega^E_k)/2&-(dt\cdot\omega^N_k)/2&1&0&0&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&1&0&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&1&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&0&1
\end{bmatrix}.
$$

*Notes:*  
$\bullet$ *We were able to write lines 10-13 of this matrix using our  observation that each component of $\mathbf q_{k+1}$ was of the form $[a(q^0_k)+b(q^1_k)+c(q^2_k)+d(q^3_k)]$, but we could have also found the form of the $n^\text{th}$ component to be $[q^n + a(\omega^N_k)+b(\omega^E_k)+c(\omega^D_k)]$. This second form will let us write a second, equavalent matrix. As an intellecual exercise, you may double check this math by building this filter with the second matrix and verifying it produces the same results.*  
$\bullet$ *This state transition matrix assumes the acceleration and angular velocity at time $t_{k+1}$ will approximately equal those at time $t_k$, which is a limitation of this filter during jerky events such as heel strike.*

### Q

The process noise covariance matrix $\mathbf Q$ represents the uncertainties in the system dynamics. For our state vector, this would look like
$$
\mathbf Q =
\begin{bmatrix}
0 & \sigma^2_{p^{\text{E}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
\sigma^2_{p^{\text{N}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & \sigma^2_{p^{\text{D}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & \sigma^2_{v^{\text{N}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & \sigma^2_{v^{\text{E}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & \sigma^2_{v^{\text{D}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{a^{\text{N}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{a^{\text{E}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{a^{\text{D}}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{q^0} & 0 & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{q^1} & 0 & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{q^2} & 0 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{q^3} & 0 & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{N}}} & 0 & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{E}}} & 0 \\\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{D}}} 
\end{bmatrix},
$$
Where: $\sigma^2_{p^\text N}$ is the variance in the north axis position, and so on and so forth. The specific values for these variances would depend on the characteristics of the system and the expected process noise.

## Measurement

The kalman filter's measurment is described by the measurement mean $\mathbf z$, and the noise covariance $\mathbf R$.

### z

As described in the *"Measurement Variable"* section, we will have
$$
\mathbf z_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \\\ \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix}.
$$

### R

Since $\mathbf z$ is a 6x1 vector, $\mathbf R$ will be a 6x6 matrix representing the noise covariance of our measurements. A reasonable $\mathbf R$ would be:
$$
\mathbf R =
\begin{bmatrix}
\sigma^2_{a^{\text{pitch}}} & 0 & 0 & 0 & 0 & 0 \\\\
0 & \sigma^2_{a^{\text{roll}}} & 0 & 0 & 0 & 0 \\\\
0 & 0 & \sigma^2_{a^{\text{yaw}}} & 0 & 0 & 0 \\\\
0 & 0 & 0 & \sigma^2_{\omega^{\text{pitch}}} & 0 & 0 \\\\
0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{roll}}} & 0\\\\
0 & 0 & 0 & 0 & 0 & \sigma^2_{\omega^{\text{yaw}}}
\end{bmatrix},
$$
where $\sigma^2_{a^{\text{pitch}}}$ is the variance in the pitch acceleration measurements, and so on and so forth. If we expect the accelerometers and gyroscopes to have the same variance in all directions, we may choose to use a single value for $\sigma^2_a$ and a single value for $\sigma^2_{\omega}$.

## Measurement Function: H

Our given forms of $\mathbf x_k$ and $\mathbf z_k$ mean we'll have some 6x16 measurement function $\mathbf H$ such that
$$
\mathbf y_k = \mathbf z_k - \left(\mathbf H \mathbf\cdot\mathbf x_k\right).
$$

To visualize this in expanded form, we want a matrix $\mathbf H$ which satisfies
$$
\mathbf y_k = \begin{bmatrix} a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\ a^{\text{yaw}}_k \\\ \omega^{\text{pitch}}_k \\\ \omega^{\text{roll}}_k \\\ \omega^{\text{yaw}}_k \end{bmatrix} -
\left(\begin{bmatrix}?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\end{bmatrix}
\begin{bmatrix} p^\text{N}_k\\\\ p^\text{E}_k\\\\ p^\text{D}_k\\\\ v^\text{N}_k\\\\ v^\text{E}_k\\\\ v^\text{D}_k\\\\ a^\text{N}_k\\\\ a^\text{E}_k\\\\ a^\text{D}_k\\\\ q^0_k\\\\ q^1_k\\\\ q^2_k\\\\ q^3_k\\\\ \omega^\text{N}_k\\\\ \omega^\text{E}_k\\\\ \omega^\text{D}_k \end{bmatrix}\right).
$$
We see that the first three rows will each be dotted with $\mathbf x_k$ to get the local acceleration, and the next three rows will be dotted with $\mathbf x_k$ to get the local rotational velocity. Using our $\mathbf C$ matrix from the *"Translating Between Local and World Frames"* section, we have
$$
\mathbf a^{\text{local}}_k = \mathbf C^T_k \begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9.8\end{bmatrix}.
$$
This expands to
$$
\begin{bmatrix}a^{\text{pitch}}_k \\\\ a^{\text{roll}}_k \\\\ a^{\text{yaw}}_k\end{bmatrix} =
\begin{bmatrix}
1 - 2\big((q^2_k)^2 + (q^3_k)^2\big) & 2\big(q^1_k q^2_k - q^0_k q^3_k\big) & 2\big(q^1_k q^3_k + q^0_k q^2_k\big) \\\\
2\big(q^1_k q^2_k + q^0_k q^3_k\big) & 1 - 2\big((q^1_k)^2 + (q^3_k)^2\big) & 2\big(q^2_k q^3_k - q^0_k q^1_k\big) \\\\
2\big(q^1_k q^3_k - q^0_k q^2_k\big) & 2\big(q^2_k q^3_k + q^0_k q^1_k\big) & 1 - 2\big((q^1_k)^2 + (q^2_k)^2\big)
\end{bmatrix}^T
\begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9.8\end{bmatrix}.
$$
To simplify things, let's define
$$
\begin{align*}
c^0_k &= 1 - 2\big((q^2_k)^2 + (q^3_k)^2\big) \\\\
c^1_k &= 2\big(q^1_k q^2_k - q^0_k q^3_k\big) \\\\
c^2_k &= 2\big(q^1_k q^3_k + q^0_k q^2_k\big) \\\\
c^3_k &= 2\big(q^1_k q^2_k + q^0_k q^3_k\big) \\\\
c^4_k &= 1 - 2\big((q^1_k)^2 + (q^3_k)^2\big) \\\\
c^5_k &= 2\big(q^2_k q^3_k - q^0_k q^1_k\big) \\\\
c^6_k &= 2\big(q^1_k q^3_k - q^0_k q^2_k\big) \\\\
c^7_k &= 2\big(q^2_k q^3_k + q^0_k q^1_k\big) \\\\
c^8_k &= 1 - 2\big((q^1_k)^2 + (q^2_k)^2\big),
\end{align*}
$$
so that we can write
$$
\begin{bmatrix}a^{\text{pitch}}_k \\\ a^{\text{roll}}_k \\\\ a^{\text{yaw}}_k\end{bmatrix} =
\begin{bmatrix}
c^0_k & c^1_k & c^2_k \\\
c^3_k & c^4_k & c^5_k \\\
c^6_k & c^7_k & c^8_k
\end{bmatrix}^T
\begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9.8\end{bmatrix}
= \begin{bmatrix}
c^0_k & c^3_k & c^6_k \\\
c^1_k & c^4_k & c^7_k \\\
c^2_k & c^5_k & c^8_k
\end{bmatrix}
\begin{bmatrix}a^{\text{N}}_k \\\ a^{\text{E}}_k \\\ a^{\text{D}}_k + 9.8\end{bmatrix}.
$$
From here, we can start to fill in the first three rows of $\mathbf H$:
$$
\mathbf H = \begin{bmatrix}
0&0&0&0&0&0&c^0_k&c^3_k&\left(c^6_k + 9.8c^6_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&c^1_k&c^4_k&\left(c^7_k + 9.8c^7_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&c^2_k&c^5_k&\left(c^8_k + 9.8c^8_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?\\\\
?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?
\end{bmatrix}.
$$

The bottom three rows will be quite similar. In order to find them, we will use
$$
\boldsymbol\omega^{\text{local}}_k = \mathbf C^T_k \cdot \boldsymbol\omega^{\text{world}}_k,
$$
Which expands to 
$$
\begin{bmatrix}\omega^{\text{pitch}}_k \\\\ \omega^{\text{roll}}_k \\\\ \omega^{\text{yaw}}_k\end{bmatrix} =
\begin{bmatrix}
c^0_k & c^3_k & c^6_k \\\\
c^1_k & c^4_k & c^7_k \\\\
c^2_k & c^5_k & c^8_k
\end{bmatrix}
\begin{bmatrix}\omega^{\text N}_k \\\\ \omega^{\text E}_k \\\\ \omega^{\text D}_k\end{bmatrix},
$$
Meaning our matrix should look like
$$
\mathbf H =
\begin{bmatrix}
0&0&0&0&0&0&c^0_k&c^3_k&\left(c^6_k + 9.8c^6_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&c^1_k&c^4_k&\left(c^7_k + 9.8c^7_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&c^2_k&c^5_k&\left(c^8_k + 9.8c^8_k/a_k^\text D\right)&0&0&0&0&0&0&0\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&c^0_k&c^3_k&c^6_k\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&c^1_k&c^4_k&c^7_k\\\\
0&0&0&0&0&0&0&0&0&0&0&0&0&c^2_k&c^5_k&c^8_k
\end{bmatrix}.
$$

from AdriensFunctions import *
from random import *

"""
https://www.youtube.com/watch?v=HCd-leV8OkU

Notes from "Kalman Filter for Beginners, Part 1 - Recursive Filters & MATLAB Examples"
____________________________________________________________________________________________________
Recursive expression for average:

definition of average at k:                     xbar_k = (x1 + x2 + x3 + ... + xk)/k           (a)
definition of average at k-1:                 xbar_k-1 = (x1 + x2 + x3 + ... + xk-1)/(k-1)     (b) 
multiply eq. (a) by k:                      k * xbar_k = (x1 + x2 + x3 + ... + xk)             (c) 
multiply eq. (b) by (k-1) and substitute:   k * xbar_k = (k-1) * xbar_k-1 + xk                 (d)
divide by k for the desired:                    xbar_k = (k-1)/k * xbar_k-1 + 1/k * xk 
                                                        |_______|            |___|
                                                            |                  |
                                                            |      as k increases, this approaches 0
                                                            |
                                              as k increases, this approaches 1

Note: since 1-(k-1)/k simplifies to 1/k, we can set alpha = (k-1)/k and rewrite this as:
                                                xbar_k = alpha * xbar_k-1 + (1-alpha) * xk
____________________________________________________________________________________________________
First order low pass filter equation:

for 0 < alpha < 1:      xbar_k = alpha * xbar_k-1 + (1-alpha) * xk

Note: alpha close to 0 weights new data more
____________________________________________________________________________________________________
Kalman filter equation steps:

0. set initial values:                 xhat_0, P_0
1. predict state and error covariance: xhatminus_k = A * xhat_k-1
                                          Pminus_k = A * P_k-1 * A^T + Q
2. compute Kalman gain:                        K_k = Pminus_k * H^T * (H * Pminus_k * H^T + R)^-1
3. compute the estimate w/ measurment z_k:  xhat_k = xhatminus_k + K_k * (z_k - H * xhatminus_k)
4. compute the error covariance                P_k = Pminus_k - K_k * H * Pminus_k

Note 1.: 'minus' -> before measurement
xhatminus_k = predicted state   Pminus_k = predicted error covariance

Note 2. & 3.: 'estimation step' is computed using matricies     Q: does K_k = Kalman gain?

          external input ---> z_k
            final output ---> xhat_k (estimate)
            system model ---> A, H, Q, R
for internal computation ---> xhatminus_k, Pminus_k, P_k, K_k

A: matrix estimated linear model of process - prediction step
Q: process noise / state transition noise   - prediction step
H: describes how measurements map to states - estimation step
R: represent noise levels                   - estimation step

x_k   = state variable            (nx1 column vector)
z_k   = measurment                (mx1 column vector)
A =   state transition matrix     (nxm matrix)
H   = state-to-measurement matrix (mxn matrix)
w_k = state transition noise      (nx1 column vector)
v_k = measurment noise            (mx1 column vector)

Q = covariace matrix of w_k (nxn diagonal matrix) --- np.fill_diagonal
R = covariance matrix of v_k (mxm diagonal matrix)


"""
# end of notes


""" first order low pass filter  """
def LPF(data, alpha, initial = 'NA'):
    # no initial value
    if initial == 'NA':
        estimates = [data[0]]
        for prev, measurment in enumerate(data[1:]):
            estimates.append(alpha*estimates[prev] + (1-alpha)*measurment) 
        return estimates
    #using initial value
    else:
        estimates = [initial]
        for prev, measurment in enumerate(data):
            estimates.append(alpha*estimates[prev] + (1-alpha)*measurment) 
        return estimates[1:]


""" Basic Kalman Filter """
def Simple_Kalman(
    measurments, initial_state=0, initial_error_covariance=6, state_transition=1.03,
    measurement_model=1, process_noise=0, measurement_covariance=1
):
    # set variables
    A = state_transition
    H = measurement_model
    Q = process_noise
    R = measurement_covariance
    P = initial_error_covariance
    x = initial_state
    filtered_data = []
    # kalman algorithm
    #consider variable for A^T and H^T
    for z in measurments:
        xp = A * x                                                              # prediction of estimate
        Pp = A * P * np.transpose(A) + Q                                        # prediction of error cov
        K = Pp * np.transpose(H) * (1/(H * Pp * np.transpose(H) + R))           # kalman gain
        x = xp + K * (z - H * xp)                                               # state estimate
        P = Pp - K * H * Pp                                                     # error cov
        filtered_data.append(x)
    return filtered_data


def noisy_data(value, length, noise):
    data = [sin(i/20) for i in range(length)]
    for i, point in enumerate(data):
        data[i] += uniform(-1*noise, noise)
    return data

""" main program """
def main():
    result = Simple_Kalman([1,2,3,4,5,6,7,8,9])
    noisy = noisy_data(0, 1000, 0.5)
    noise = noisy_data(0, 1000, 0)
    noisyk = Simple_Kalman(noisy)
    plt.plot(noisy)
    plt.plot(noise)
    plt.plot(noisyk)
    plt.show()
    #print(noisy)


if __name__ == "__main__":
    main()

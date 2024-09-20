from AdriensFunctions import *

"""
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

"""



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
        return estimates

def main():
    result = LPF([1,2,3,4,5,6,7,8,9],0.9) 
    print(result)
if __name__ == "__main__":
    main()

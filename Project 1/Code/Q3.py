import numpy as np
import math

# Name: Tien Li Shen
# Homework 1 Q3

def pt1(F, S):
    # R = [[Rr1, Rr2, Rr3],
    #      [Rg1, Rg2, Rg3],
    #      [Rb1, Rb2, Rb3]]
    size = F.shape[0]
    R = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            R[i][j] = np.dot(F[j], S[i])
    return R

def pt2(R):
    C_turquoise = np.array([0.2509, 0.8784, 0.8156])
    C_goldenrod = np.array([0.8549, 0.6470, 0.1254])

    # turquoise
    R_inv = np.linalg.inv(R)
    b_turquoise = np.zeros(C_turquoise.shape[0])
    for i in range(C_turquoise.shape[0]):
        b_turquoise[i] = np.dot(R_inv[i], C_turquoise)

    # goldenrod
    b_goldenrod = np.zeros(C_goldenrod.shape[0])
    for i in range(C_goldenrod.shape[0]):
        b_goldenrod[i] = np.dot(R_inv[i], C_goldenrod)

    return [b_turquoise, b_goldenrod]


if __name__ == '__main__':

    # F => 3 flash lights emssion spectral distribution
    F = np.array([[0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.07, 0.29, 0.35, 0.12],
    [0.00, 0.01, 0.02, 0.06, 0.20, 0.31, 0.20, 0.16, 0.04, 0.00],
    [0.03, 0.15, 0.25, 0.27, 0.12, 0.02, 0.01, 0.01, 0.00, 0.00]])

    # S => Human eye RGB absorption distribution
    S = np.array([[0.16, 0.26, 0.28, 0.15, 0.10, 0.03, 0.02, 0.00, 0.00, 0.00],
    [0.00, 0.03, 0.06, 0.20, 0.31, 0.21, 0.15, 0.03, 0.01, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.01, 0.04, 0.08, 0.23, 0.35, 0.29]])

    R = pt1(F, S)
    print("PT1\nR = ", R)
    b_turquoise, b_goldenrod = pt2(R)
    print("PT2\nb_turquoise = {}\nb_goldenrod = {}".format(b_turquoise, b_goldenrod))





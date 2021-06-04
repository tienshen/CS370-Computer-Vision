#This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2021
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Mini-Project 4


import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from nms import nms

from scipy.ndimage import gaussian_filter


def detectCorners(I, is_simple, w, th):
#Convert to float
    I = I.astype(float)

    #Convert color to grayscale
    if I.ndim > 2:
        I = rgb2gray(I)

    # Step 1: compute corner score
    if is_simple:
        corner_score = simple_score(I, w)
    else:
        corner_score = harris_score(I, w)

    # Step 2: Threshold corner score and find peaks
    corner_score[corner_score < th] = 0

    cx, cy, cs = nms(corner_score)
    return cx, cy, cs, corner_score


#--------------------------------------------------------------------------
#                                    Simple score function (Implement this)
#--------------------------------------------------------------------------
def simple_score(I, w):
    corner_score = np.zeros(I.shape)
    # E(u; v) = (I * f(u; v))^2 * Gsigma
    # use a nested for loop to find the gradient
    for i in range(1, I.shape[0]-1):
        for j in range(1, I.shape[1]-1):
            for u in range(-1, 1):
                for v in range(-1, 1):
                    corner_score[i][j] += I[i+u][j+v] - I[i][j]
    corner_score = corner_score**2
    corner_score = gaussian_filter(corner_score, sigma = w)

    return corner_score


#--------------------------------------------------------------------------
#                                    Harris score function (Implement this)
#--------------------------------------------------------------------------
def harris_score(I, w):
    k = 0.04  # as suggested by the assignment pdf
    corner_score = np.zeros(I.shape)
    # get first derivative
    ix, iy = gradient(I)

    # get second order derivative and gaussian filter the second order derivative
    ixx = gaussian_filter(ix ** 2, sigma=w)
    ixy = gaussian_filter(ix * iy, sigma=w)
    iyy = gaussian_filter(iy ** 2, sigma=w)

    # lamda1 * lambd2 = det(M) = ad - bc, and lambda1 + lambda2 = trace(M) = a + d. Thus you can compute the score as,
    # cornerScore = (ad - bc) - k(a + d)2:
    corner_score = ixx*ixy - iyy*ixy - k*((ixx+iyy)**2)
    return corner_score

def gradient(im):
    gx = np.zeros(im.shape)
    gy = np.zeros(im.shape)
    for i in range(1, im.shape[0]-1):
        for j in range(1, im.shape[1]-1):
            gx[i][j] = np.dot(im[i][j - 1:j + 2], [-1, 0, 1])  # get the gradient in the x direction
            gy[i][j] = np.dot([im[i-1][j], im[i][j], im[i+1][j]], [-1, 0, 1]) # get the gradient in the y direction
    return [gx, gy]


# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2021
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 2

import numpy as np
from randomlyShiftChannels import randomlyShiftChannels
from PIL import Image


def findTheta(img_arr, channel, i, j, max_shift):
    # computes the angle of 2 vectors, image matrix is flattened to get vector
    if img_arr.shape[0] > 10*max_shift[0] and img_arr.shape[1] > 10*max_shift[1]:
        img_arr = img_arr[max_shift[0]*2:-max_shift[0]*2][max_shift[1]*2:-max_shift[1]*2] # crop image
    img_channel = np.roll(img_arr[:, :, channel], [i, j], axis=[0, 1]).flatten()  # shift channel
    theta = np.arccos(np.matmul(np.transpose(img_channel), img_arr[:, :, 0].flatten()) / (np.linalg.norm(img_channel) * np.linalg.norm(img_arr[:, :, 0])))
    # print(theta)
    return theta


def alignChannels(img, max_shift):

    img_arr = np.array(img)
    channels = np.array([1, 2])
    lowest_theta_shift = np.zeros((2,2))

    # find offset lowest_theta_shift
    for c in channels:
        lowest_theta = 1000
        for i in range(-max_shift[0], max_shift[0]):
            for j in range(-max_shift[1], max_shift[1]):
                theta = findTheta(img_arr, c, i, j, max_shift)
                if theta < lowest_theta:
                    lowest_theta = theta
                    lowest_theta_shift[c-1] = [i , j]

        # allign the image with coordinates
        img_arr[:, :, c] = np.roll(img_arr[:, :, c], np.intc(lowest_theta_shift[c-1]), axis=[0, 1])

    # convert numpy image array to Image
    return [img_arr, lowest_theta_shift]

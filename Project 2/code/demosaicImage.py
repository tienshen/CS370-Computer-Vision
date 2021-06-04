# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2021
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 2

import numpy as np

def demosaicImage(image, method):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''

    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[0:image_height:2, 0:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 0] = img[0:image_height:2, 0:image_width:2]

    blue_values = img[1:image_height:2, 1:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 2] = img[1:image_height:2, 1:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of size NxMx3 demosaiced using the nearest neighbor 
        algorithm.
    '''

    image_height, image_width = img.shape
    demos_img = np.zeros((image_height, image_width, 3))

    #iterate through the odd rows
    for i in range(0, image_height, 2):
        # red pixels
        for j in range(0, image_width, 2):
            if j+1 < image_width and i+1 < image_height: # in the situation of a non-edge/corner pixel
                demos_img[i][j] = [img[i][j], img[i][j + 1], img[i + 1][j]]
            elif j+1 >= image_width and i+1 >= image_height: # in the situation of a corner pixel
                demos_img[i][j] = [img[i][j], img[i][j - 1], img[i - 1][j]]
            elif j+1 >= image_width: # in the situation of a right edge pixel
                demos_img[i][j] = [img[i][j], img[i][j - 1], img[i + 1][j]]
            else: # in situation of a bottom edge pixel
                demos_img[i][j] = [img[i][j], img[i][j + 1], img[i - 1][j]]
        # green pixels
        for j in range(1, image_width, 2):
            if j+1 < image_width and i+1 < image_height: # in the situation of a non-edge/corner pixel
                demos_img[i][j] = [img[i][j + 1], img[i][j], img[i + 1][j]]
            if j+1 >= image_width and i+1 >= image_height: # in the situation of a corner pixel
                demos_img[i][j] = [img[i][j - 1], img[i][j], img[i - 1][j]]
            elif j+1 >= image_width: # in the situation of a right edge pixel
                demos_img[i][j] = [img[i][j - 1], img[i][j], img[i + 1][j]]
            else: # in situation of a bottom edge pixel
                demos_img[i][j] = [img[i][j + 1], img[i][j], img[i - 1][j]]

    # iterate through the even rows
    for i in range(1, image_height, 2):
        # green pixels
        for j in range(0, image_width, 2):
            if j+1 < image_width and i+1 < image_height: # in the situation of a non-edge/corner pixel
                demos_img[i][j] = [img[i - 1][j], img[i][j], img[i][j + 1]]
            if j+1 >= image_width and i+1 >= image_height: # in the situation of a corner pixel
                demos_img[i][j] = [img[i - 1][j], img[i][j], img[i][j - 1]]
            elif j+1 >= image_width: # in the situation of a right edge pixel
                demos_img[i][j] = [img[i - 1][j], img[i][j], img[i][j - 1]]
            else: # in situation of a bottom edge pixel
                demos_img[i][j] = [img[i - 1][j], img[i][j], img[i][j + 1]]
        # blue pixels
        for j in range(1, image_width, 2):
            if j+1 < image_width and i+1 < image_height: # in the situation of a non-edge/corner pixel
                demos_img[i][j] = [img[i + 1][j], img[i][j + 1], img[i][j]]
            if j+1 >= image_width and i+1 >= image_height: # in the situation of a corner pixel
                demos_img[i][j] = [img[i - 1][j], img[i][j - 1], img[i][j]]
            elif j+1 >= image_width: # in the situation of a right edge pixel
                demos_img[i][j] = [img[i + 1][j], img[i][j - 1], img[i][j]]
            else: # in situation of a bottom edge pixel
                demos_img[i][j] = [img[i - 1][j], img[i][j + 1], img[i][j]]

    return demos_img



# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2021
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 2

#   Name: Tien Li Shen
#   Date: 3/7/2021

import numpy as np

def mosaicImage(img):
    ''' Computes the mosaic of an image.

    mosaicImage computes the response of the image under a Bayer filter.

    Args:
        img: NxMx3 numpy array (image).

    Returns:
        NxM image where R, G, B channels are sampled according to RGRG in the
        top left.
    '''

    image_height, image_width, num_channels = img.shape

    assert(num_channels == 3) #Checks if it is a color image

    color_img = np.zeros((image_height, image_width))
    color_img[::2] = [[i[j][j%2] for j in range(image_width)] for i in img[::2]] # odd rows
    color_img[1::2] = [[i[j][j%2+1] for j in range(image_width)] for i in img[1::2]] # even rows

    return color_img

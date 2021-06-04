import numpy as np
from skimage.transform import resize
from scipy import ndimage
import matplotlib.pyplot as plt
import utils
from PIL import Image


def vis_hybrid_image(hybrid_image):
    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]
    output = hybrid_image.copy()
    cur_image = hybrid_image.copy()

    for i in range(1, scales):
        output = np.concatenate((output, np.ones((original_height, padding, num_colors))), axis=1)

        cur_image = resize(cur_image, (int(scale_factor*cur_image.shape[0]),
            int(scale_factor*cur_image.shape[1])))

        tmp = np.concatenate((np.ones((original_height - cur_image.shape[0], cur_image.shape[1], 
            num_colors)), cur_image), axis=0)

        output = np.concatenate((output, tmp), axis=1)

    return output


#Function test
def hybridImage(im1, im2, sigma1, sigma2):
    """
    input: 2 colored images, and sigmas for the gaussian filter

    output: the hybrid image

        hybrid image = blurry(I) + sharp(I):

        Ihybrid = blurry(I1; sigma1) + sharp(I2; sigma2) = I1 conv g(sigma) + I2 * I2 conv g(sigma2)

    """

    im1_gau = ndimage.gaussian_filter(im1, sigma1) # apply gaussian filter with sigma1 to image1 and store it in variable im1_gau
    im2_gau = ndimage.gaussian_filter(im2, sigma2) # apply gaussian filter with sigma2 to image2 and store it in variable im2_gau
    hybrid_im = im1_gau + im2 - im2_gau # apply the equation: hybrid_image = I1 conv g(sigma) + I2 * I2 conv g(sigma2)
    hybrid_im = np.clip(hybrid_im, a_min = 0, a_max = 1) # clipping image to min of 0 and max of 1

    return hybrid_im


if __name__ == '__main__':
    # img = utils.imread('../data/dog.jpg')
    # plt.imshow(vis_hybrid_image(img))
    # plt.show()

    # import images and declare hyperparmeters
    im1 = utils.imread('../data/jotaro.jpg')
    im2 = utils.imread('../data/minutemen.jpg')
    sigma1, sigma2 = [2, 11]

    # scale,shit, and crop image correctly
    scale_factor = 1.3
    im1 = resize(im1, (int(1/scale_factor * im1.shape[0]), int(1/scale_factor * im1.shape[1]))) # scale image
    #crop image
    x = 115
    y = 30
    im1 = im1[x:225+x,y:300+y,:]

    hybrid_im = hybridImage(im1, im2, sigma1, sigma2) # get hybrid image
    plt.imshow(vis_hybrid_image(hybrid_im))
    plt.show()

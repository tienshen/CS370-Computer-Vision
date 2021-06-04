import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def load_image():
    im1 = plt.imread('../data/peppers.png')
    im2 = plt.imread('../data/peppers_g.png')
    im3 = plt.imread('../data/peppers_sp.png')

    im_123 = np.array([im1, im2, im3])

    return im_123

def gaussian_filter(im_123, sigma):
    plt.figure(1)
    count = 0

    # apply gaussian filter to Gaussian noise image
    error_g = np.zeros(len(sigma))
    for s in sigma: # use for loop to use gaussian filter on each sigma tested
        im_gau = ndimage.gaussian_filter(im_123[1], sigma = s) # use scipy gaussian filter on image
        error_g[count] = ((im_123[0] - im_gau) ** 2).sum()  # calculate the error

        # configure the plot
        plt.subplot(1, len(sigma), count+1)
        plt.title("Gaussian noise (s={})".format(s))
        plt.imshow(im_gau, cmap = "gray")
        plt.axis('off')
        count+=1

    # apply gaussian filter to pepper noise image
    plt.figure(2)
    count = 0
    error_sp = np.zeros(len(sigma))

    for s in sigma: # use for loop to use gaussian filter on each sigma tested
        im_gau = ndimage.gaussian_filter(im_123[2], sigma=s) # use scipy gaussian filter on image
        error_sp[count] = ((im_123[0] - im_gau) ** 2).sum() # calculate the error

        # configure the plot
        plt.subplot(1, len(sigma), count+1)
        plt.title("Pepper noise (s={})".format(s))
        plt.imshow(im_gau, cmap="gray")
        plt.axis('off')
        count += 1
    print(sigma)
    print(error_g)
    print(error_sp)
    plt.show()

def median_filter(im_123, sigma):
    plt.figure(1)
    count = 0

    # apply gaussian filter to Gaussian noise image
    error_g = np.zeros(len(sigma))
    for s in sigma: # use for loop to use median filter on each size tested
        im_gau = ndimage.median_filter(im_123[1], size=s)  # use scipy median filter on image
        error_g[count] = ((im_123[0] - im_gau) ** 2).sum()  # calculate the error

        # configure the plot
        plt.subplot(2, int(len(sigma) / 2), count + 1)
        plt.title("Gaussian noise (s={})".format(s))
        plt.imshow(im_gau, cmap="gray")
        plt.axis('off')
        count += 1

    # apply gaussian filter to pepper noise image
    plt.figure(2)
    count = 0
    error_sp = np.zeros(len(sigma))
    for s in sigma: # use for loop to use median filter on each size tested
        im_gau = ndimage.median_filter(im_123[2], size=s) # use scipy median filter on image
        error_sp[count] = ((im_123[0] - im_gau) ** 2).sum()  # calculate the error

        # configure the plot
        plt.subplot(2, int(len(sigma)/2), count + 1)
        plt.title("Pepper noise(s={})".format(s))
        plt.imshow(im_gau, cmap="gray")
        plt.axis('off')
        count += 1
    print(sigma)
    print(error_g)
    print(error_sp)
    plt.show()

if __name__ == "__main__":
    im_123 = load_image()
    sigma = [i for i in range(1,9)]
    # sigma = [i for i in range(4)]
    median_filter(im_123, sigma)
    # gaussian_filter(im_123, sigma)
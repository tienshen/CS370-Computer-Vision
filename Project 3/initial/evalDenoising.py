# This code is part of:
#
#   CMPSCI 370: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
from utils import imread

im = imread('../data/peppers.png')
noise1 = imread('../data/peppers_g.png')
noise2 = imread('../data/peppers_sp.png')

error1 = ((im - noise1)**2).sum()
error2 = ((im - noise2)**2).sum()

print('Input, Errors: {:.2f} {:.2f}'.format(error1, error2))

plt.figure(1)

plt.subplot(131)
plt.imshow(im, cmap="gray")
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1, cmap="gray")
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(noise2, cmap="gray")
plt.title('SE {:.2f}'.format(error2))

plt.show()

# Denoising algorithm (Gaussian filtering)

# Denoising algorithm (Median filtering)

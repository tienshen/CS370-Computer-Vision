#This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2021
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Mini-Project 4


import numpy as np
import matplotlib.pyplot as plt
import utils
from skimage import data
from detectCorners import detectCorners

# I = data.checkerboard()
# I = utils.imread('../data/polymer-science-umass.jpg')
# I = utils.imread('../data/EndGrainChessBoard.jpg')
I = utils.imread('../data/SydneyOperaHouse.jpeg')
plt.figure(1)
cx, cy, cs, corner_score = detectCorners(I, True, 0.5, 0.05)
plt.subplot(221)
#from IPython import embed; embed(); exit(-1)
if I.ndim == 2:
    plt.imshow(I, cmap='gray')
else:
    plt.imshow(I)

plt.plot(cx, cy, 'r.')
plt.title('Simple Corners')
plt.axis('off')
plt.subplot(223)
plt.imshow(corner_score, cmap='gray')
plt.title('Simple Corners (corner scores)')
plt.axis('off')

cx, cy, cs, corner_score = detectCorners(I, False, 0.5, 0.0001)
plt.subplot(222)
if I.ndim == 2:
    plt.imshow(I, cmap='gray')
else:
    plt.imshow(I)
plt.title('Harris Corners')
plt.axis('off')
plt.plot(cx, cy, 'g.')
plt.subplot(224)
plt.imshow(corner_score, cmap='gray')
plt.title('Harris Corners (corner scores)')
plt.axis('off')

plt.show()


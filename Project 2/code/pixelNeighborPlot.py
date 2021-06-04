

import matplotlib.pyplot as plt
import numpy as np

# Author: Tien Li Shen
# Date: 3/7/2021
# Class: CS370

from PIL import Image
# Open the image form working directory
image = Image.open("../data/demosaic/puppy.jpg")
img_arr = np.array(image)

length, width, channels = img_arr.shape
red_ch = img_arr[:,:,0].flatten()
gre_ch = img_arr[:,:,1].flatten()
blu_ch = img_arr[:,:,2].flatten()

plt.figure(1)
plt.clf()
plt.subplot(131)
plt.title('red channel'); plt.plot(red_ch[1:],red_ch[:-1], '.', color='red');
plt.subplot(132)
plt.title('green channel'); plt.plot(gre_ch[1:],red_ch[:-1], '.', color='green');
plt.subplot(133)
plt.title('blue channel'); plt.plot(blu_ch[1:],red_ch[:-1], '.', color='blue');
plt.show()
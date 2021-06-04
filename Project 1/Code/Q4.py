import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def grayworld(im): #[L, C] = grayworld(I)
    I = np.array(im)
    dim = I.shape
    I_avg = np.zeros(3)

    #calculate the average I_avg values
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(len(I_avg)):
                I_avg[k] += I[i][j][k]
    I_avg = I_avg/dim[0]/dim[1]

    L = I_avg/np.array([128, 128, 128])

    #get the 128/r, 128/g, 128/b ratios
    rgb_ratio = np.array([128, 128, 128]) / I_avg

    #get color image C
    C = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
                C[i][j] = I[i][j] * rgb_ratio

    # scale pixels to max of 255
    C = C/np.amax(C)*255

    C = Image.fromarray(np.uint8(C))
    return [L, C]

if __name__ == '__main__':
    im = Image.open(r"data/wb_sardmen-incorrect.jpg")

    L, C = grayworld(im)
    print("L = ", L)
    c_arr = np.array(C)
    I = c_arr * L
    I = Image.fromarray(np.uint8(I))

    # plot the image
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(C)
    plt.title("Color Corrected")
    plt.show()

    # plt.subplot(1, 2, 1)
    # plt.imshow(im)
    # plt.title("Original")
    # plt.subplot(1, 2, 2)
    # plt.imshow(I)
    # plt.title("Reconstructed Original")
    # plt.show()
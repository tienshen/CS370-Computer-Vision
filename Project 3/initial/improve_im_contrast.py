import utils
import numpy as np
import matplotlib.pyplot as plt

def histogram(im):
    """
    input: image with only 1 channel

    return: [pdf, cdf]
    """
    # count the occurance of pixel values and store them in i_arr
    i_arr = np.zeros(256)
    for row in im:
        for pix in row:
            i_arr[pix.astype(int)] += 1

    # optionally convert the PDF to the decimal probability scale
    pix_sum = sum(i_arr)
    pdf = i_arr #/pix_sum

    # declare and compute the cdf
    cdf = np.zeros(256)
    cdf[0] = pdf[0]
    count = 0
    for i in range(len(pdf)): # sum the cdf with a for loop
        count += pdf[i]
        cdf[i] = count


    return [pdf, cdf]

def contrast_stretch(im):
    # declare variables needed
    im_max = np.amax(im)
    im_min = np.amin(im)
    im_diff = im_max - im_min
    im_stretched = np.zeros(im.shape) # placeholder for converted image
    for i in range(im.shape[0]): # loop over rows
        for j in range(im.shape[1]): # loop over columns/each pixels
            im_stretched[i][j] = ((im[i][j]-np.amin(im))/(im_max - im_min) * 255).astype(np.uint8) # apply streching equation to each pixel

    return im_stretched

def compare_stretch(im):
    """
    function to use other functions and get desired results in the desired format for the assignmnet submission
    """
    im_stretched = contrast_stretch(im)
    pdf, cdf = histogram(im)
    #
    x = [i for i in range(256)]
    plt.subplot(2, 2, 1)
    plt.title("PDF (original)")
    plt.bar(x, pdf)
    plt.subplot(2, 2, 2)
    plt.title("CDF (original)")
    plt.plot(cdf)

    pdf, cdf = histogram(im_stretched)
    #
    x = [i for i in range(256)]
    plt.figure(1)
    plt.subplot(2, 2, 3)
    plt.title("PDF (stretched)")
    plt.bar(x, pdf)
    plt.subplot(2, 2, 4)
    plt.title("CDF (stretched)")
    plt.plot(cdf)

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap="gray")
    plt.title("original")
    plt.subplot(1, 2, 2)
    plt.imshow(im_stretched, cmap="gray")
    plt.title("stretched")
    plt.show()

def gamma_correction(im, r):
    """
    apply gamma correction with the provided equation in the assignment PDF
    """
    im_arr = np.array(im)
    im_arr = (im_arr / 255)**r * 255

    return im_arr
def compare_gamma(im):
    """
    function to use other functions and get desired results in the desired format for the assignmnet submission
    """
    gamma_corrected = gamma_correction(im, 2)
    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap="gray")
    plt.title("original")
    plt.subplot(1, 3, 2)
    plt.imshow(gamma_corrected, cmap="gray")
    plt.title("r = 2")
    plt.subplot(1, 3, 3)
    gamma_corrected = gamma_correction(im, 0.5)
    plt.imshow(gamma_corrected, cmap="gray")
    plt.title("r = 0.5")
    plt.show()

def hist_equalization(cdf, im):
    """
    input: cdf, im

    output: equalize image

    equalize the image by applying cdf equalization to the image, the image can then be used to obtain a equalized cdf
    """
    N = im.flatten().shape[0]
    print(N)
    print(im.shape)
    count = 1
    cdf_min = 0
    im_eq = np.zeros(im.shape)
    # use a while loop to find the smallest non-zero cdf value (the first value that is non-zero)
    while(cdf[count-1] <= 0):
        cdf_min = cdf[count]
        count+=1

    # use nested for loop to apply cdf equalization to each pixel
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im_eq[i][j] = ((cdf[im[i][j]] - cdf_min) / (N - cdf_min) * 255).astype("int")

    # cdf_eq = ((cdf - cdf_min) / (N - cdf_min) * 255).astype(int)
    return im_eq

def compare_hist_equalization(im):
    """
    function to use other functions and get desired results in the desired format for the assignmnet submission
    """
    pdf, cdf = histogram(im)
    im_eq = hist_equalization(cdf, im)
    pdf_eq,  cdf_eq = histogram(im_eq)

    plt.subplot(1, 4, 1)
    plt.bar([i for i in range(256)],pdf)
    plt.title("original pdf")
    plt.subplot(1, 4, 2)
    plt.plot(cdf)
    plt.title("original cdf")
    plt.subplot(1, 4, 3)
    plt.bar([i for i in range(256)], pdf_eq)
    plt.title("equalized pdf")
    plt.subplot(1, 4, 4)
    plt.plot(cdf_eq)
    plt.title("equalized cdf")
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap="gray")
    plt.title("original cdf")
    plt.subplot(1, 2, 2)
    plt.imshow(im_eq, cmap="gray")
    plt.title("equalized cdf")
    plt.show()

if __name__ == '__main__':
    """
    Call the functions to generate output plots and images:
    
        compare_stretch(im)
        compare_gamma(im)
        compare_hist_equalization(im)
        
    """
    im = plt.imread('../data/forest.png')
    im *= 255
    im = im.astype(np.uint8)
    compare_hist_equalization(im)



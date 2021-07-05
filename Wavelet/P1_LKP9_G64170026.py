import numpy as np
from matplotlib import pyplot as plt
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
import cv2

# Load image
img1 = cv2.imread("fusarium-patogen-cropped.png")
img2 = cv2.imread("culvularia-patogen-cropped.png")

# Smoothing
blurred_fusarium = cv2.GaussianBlur(img1,(7,7),0)
blurred_culvularia = cv2.GaussianBlur(img2,(7,7),0)

# Grayscale Convertion
fusarium_gray = cv2.cvtColor(blurred_fusarium, cv2.COLOR_BGR2GRAY)
culvularia_gray = cv2.cvtColor(blurred_culvularia, cv2.COLOR_BGR2GRAY)

plt.subplot(221), plt.imshow(blurred_fusarium), plt.title("Fusarium Gaussian Kernel 5*5")
plt.subplot(222), plt.imshow(blurred_culvularia), plt.title("Culvularia Gaussian Kernel 5*5")
plt.subplot(223), plt.imshow(fusarium_gray), plt.title("Fusarium Grayscale")
plt.subplot(224), plt.imshow(culvularia_gray), plt.title("Culvularia Grayscale")
plt.show()

# Otsu's thresholding with Gaussian filtering
ret1,mask1 = cv2.threshold(fusarium_gray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
cv2.imshow("Fusarium's Mask", mask1)

# Segmentasi Fusarium
segmented_fusarium = cv2.bitwise_and(img1, img1, mask = mask1)
cv2.imshow("Fusarium", segmented_fusarium)

# Otsu's thresholding with Gaussian filtering
ret2,mask2 = cv2.threshold(culvularia_gray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
cv2.imshow("Culvularia's Mask", mask2)

# Segmentasi Culvularia
segmented_culvularia = cv2.bitwise_and(img2, img2, mask = mask2)
cv2.imshow("Culvularia", segmented_culvularia)

fusarium_gray = cv2.cvtColor(segmented_fusarium, cv2.COLOR_BGR2GRAY)
culvularia_gray = cv2.cvtColor(segmented_culvularia, cv2.COLOR_BGR2GRAY)

fusarium_shape = img1.shape
culvularia_shape = img2.shape

max_lev = 3       # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots

def dwtwithLevel(image,shape):
    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(image, cmap=plt.cm.gray)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue

        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                         label_levels=label_levels)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))

        # compute the 2D DWT
        c = pywt.wavedec2(image, 'db2', mode='periodization', level=level)

        # Normalisasi masing-masing koefisien pada array secara independen
        # untuk hasil yang lebih baik
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d / np.abs(d).max() for d in c[detail_level + 1]]
        # menampilkan hasil koefisien yang telah dinormalisasi
        arr, slices = pywt.coeffs_to_array(c)

        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()

    plt.tight_layout()
    plt.show()


titles = ['LL', 'LH',
          'HL', 'HH']

def dwtFrequency(image) :
    coeffs2_fusarium = pywt.dwt2(image, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2_fusarium
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(221), plt.plot(LL, color='blue'), plt.title("LL")
    plt.subplot(222), plt.plot(LH, color='blue'), plt.title("LH")
    plt.subplot(223), plt.plot(HL, color='blue'), plt.title("HL")
    plt.subplot(224), plt.plot(HH, color='blue'), plt.title("HH")
    plt.show()

dwtwithLevel(fusarium_gray,fusarium_shape)
dwtFrequency(fusarium_gray)
dwtwithLevel(culvularia_gray,culvularia_shape)
dwtFrequency(culvularia_gray)

cv2.waitKey()
cv2.destroyAllWindows()
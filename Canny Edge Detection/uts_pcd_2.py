import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

img = cv2.imread("IPB_Exam2.jpg",0)

#---------------------------------------------------------

# 1. Gaussian Blur
# gaussian blur dengan kernel 5*5
blur_5 = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow("Gaussian Blur 5*5", blur_5)
# gaussian blur dengan kernel 7*7
blur_7 = cv2.GaussianBlur(img,(7,7),0)
cv2.imshow("Gaussian Blur 7*7", blur_7)

#---------------------------------------------------------

#2. Gradient Calculation
def sobel_filter(img):
    sobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    sobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    sobelX_abs = np.absolute(sobelX)
    sobelY_abs = np.absolute(sobelY)

    sobelX_normed = sobelX_abs / sobelX_abs.max() * 255
    sobelY_normed = sobelY_abs / sobelY_abs.max() * 255

    gradient = np.hypot(sobelX_normed, sobelY_normed)
    gradient = np.uint8(gradient / gradient.max() * 255)
    theta = np.rad2deg(np.arctan2(sobelY, sobelX))
    return (gradient, theta)

# Gradient Calculation 5*5
gradIntensity5, theta5 = sobel_filter(blur_5)
cv2.imshow('Gradient Image 5*5', gradIntensity5)
print("Gradient Calculation 5*5 :\n", gradIntensity5)
print("Theta (Slope) 5*5 :\n", theta5)
# Gradient Calculation 7*7
gradIntensity7, theta7 = sobel_filter(blur_7)
cv2.imshow('Gradient Image 7*7', gradIntensity7)
print("Gradient Calculation 7*7 :\n", gradIntensity7)
print("Theta (Slope) 7*7 :\n", theta7)
#---------------------------------------------------------

#3. Non Maximum Suppression
def non_max_suppression(img, theta):
    row, col = img.shape
    nms = np.zeros((row, col), dtype=np.uint8)

    for i in range(1, row - 1):
        for j in range(1, col - 1):
                # angle 0
                if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180) or (-22.5 <= theta[i, j] < 0) or (
                        -180 <= theta[i, j] < -157.5):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= theta[i, j] < 67.5) or (-157.5 <= theta[i, j] < -112.5):
                    q = img[i + 1, j + 1]
                    r = img[i - 1, j - 1]
                # angle 90
                elif (67.5 <= theta[i, j] < 112.5) or (-112.5 <= theta[i, j] < -67.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= theta[i, j] < 157.5) or (-67.5 <= theta[i, j] < -22.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    nms[i, j] = img[i, j]
                else:
                    nms[i, j] = 0

    return nms

# Non Max Suppression Kernel 5*5
nms5 = non_max_suppression(gradIntensity5, theta5)
cv2.imshow("Non Max Suppression 5*5", nms5)
# Non Max Suppression Kernel 7*7
nms7 = non_max_suppression(gradIntensity7, theta7)
cv2.imshow("Non Max Suppression 7*7", nms7)

#---------------------------------------------------------

# 4. Double Threshold
def threshold(image, lowThresholdRatio=0.09, highThresholdRatio=0.13):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    row, col = img.shape
    new_image = np.zeros((row, col), dtype=np.uint8)

    weak = np.uint8(100)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(image > highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)

    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

    new_image[strong_i, strong_j] = strong
    new_image[weak_i, weak_j] = weak
    return (new_image, weak, strong)

# Double Threshold Kernel 5*5
dt5, weak5, strong = threshold(nms5)
cv2.imshow("Double Threshold 5*5", dt5)
# Double Threshold Kernel 7*7
dt7, weak7, strong = threshold(nms7)
cv2.imshow("Double Threshold 7*7", dt7)

#---------------------------------------------------------

# 5. Hysteresis
def hysteresis(img, weak, strong=255):
    row, col = img.shape
    for i in range(1, row-1):
        for j in range(1, col-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

# Hysteresis Kernel 5*5
hyst5 = hysteresis(dt5,weak5)
cv2.imshow("Hysteresis 5*5", hyst5)
# Hsteresis Kernel 7*7
hyst7 = hysteresis(dt7,weak7)
cv2.imshow("Hysteresis 7*7", hyst7)

#--------------------------------------------------------

# Perbandingan Canny dengan Laplacian
canny = cv2.Canny(img, 100, 255)
cv2.imshow("canny", canny)

# read the image in gray scale
img2 = cv2.imread('IPB_Exam2.jpg',0)

# apply gaussian blur
blur_img5 = cv2.GaussianBlur(img2, (5, 5), 0)
blur_img7 = cv2.GaussianBlur(img2, (7, 7), 0)

# Positive Laplacian Operator
laplacian5 = cv2.Laplacian(blur_img5, cv2.CV_64F)
laplacian7 = cv2.Laplacian(blur_img7, cv2.CV_64F)

plt.subplot(121)
plt.imshow(laplacian5, cmap = 'gray',interpolation = 'bicubic')
plt.title('Laplacian with GaussianBlur 5*5')

plt.subplot(122)
plt.imshow(laplacian7, cmap = 'gray',interpolation = 'bicubic')
plt.title('Laplacian with GaussianBlur 7*7')

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()

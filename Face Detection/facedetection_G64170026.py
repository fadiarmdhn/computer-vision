import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("FACE DETECTION.png")

#----- YCrCb -----#

# menerapkan threshold ke gambar HSV untuk menentukan range atas dan bawah
# batas bawah ke-1
min_YCrCb1 = np.array([0,133,77],np.uint8)
# batas atas ke-1
max_YCrCb1 = np.array([235,173,127],np.uint8)
# batas bawah ke-2
min_YCrCb2 = np.array([0,135,85],np.uint8)
# batas atas ke-2
max_YCrCb2 = np.array([255,180,135],np.uint8)
# batas bawah ke-3
min_YCrCb3 = np.array([0,151,101],np.uint8)
# batas atas ke-3
max_YCrCb3 = np.array([255,199,149],np.uint8)

# konversi image dari RGB menjadi YCrCb
imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)

# membuat mask dengan
# mengembalikan nilai biner image menggunakan range bawah dan atas (min, max)
skinRegionYCrCb1 = cv2.inRange(imageYCrCb,min_YCrCb1,max_YCrCb1)
skinRegionYCrCb2 = cv2.inRange(imageYCrCb,min_YCrCb2,max_YCrCb2)
skinRegionYCrCb3 = cv2.inRange(imageYCrCb,min_YCrCb3,max_YCrCb3)

# menggabungkan gambar asli dengan mask skinRegionYCrCb
# mask berada di atas gambar asli
faceYCrCb1 = cv2.bitwise_and(image, image, mask = skinRegionYCrCb1)
faceYCrCb2 = cv2.bitwise_and(image, image, mask = skinRegionYCrCb2)
faceYCrCb3 = cv2.bitwise_and(image, image, mask = skinRegionYCrCb3)

# range ke-1
cv2.imshow("mask_YCrCb_1", skinRegionYCrCb1)
cv2.imshow("Face YCrCb_1", faceYCrCb1)
# range ke-2
cv2.imshow("mask_YCrCb_2", skinRegionYCrCb2)
cv2.imshow("Face YCrCb_2", faceYCrCb2)
# range ke-3
cv2.imshow("mask_YCrCb_3", skinRegionYCrCb3)
cv2.imshow("Face YCrCb_3", faceYCrCb3)


#----- HSV -----#
# menerapkan threshold ke gambar HSV untuk menentukan range atas dan bawah

# batas bawah ke-1
min_HSV1 = np.array([0, 58, 30],np.uint8)
# batas atas ke-1
max_HSV1 = np.array([33, 255, 255],np.uint8)
# batas bawah ke-2
min_HSV2 = np.array([0, 15, 0],np.uint8)
# batas atas ke-2
max_HSV2 = np.array([17, 170, 255],np.uint8)
# batas bawah ke-3
min_HSV3 = np.array([0, 40, 0],np.uint8)
# batas atas ke-3
max_HSV3 = np.array([25, 255, 255],np.uint8)

# konversi image dari BGR menjadi HSV
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# membuat mask dengan
# mengembalikan nilai biner image menggunakan range bawah dan atas (min, max)
skinRegionHSV1 = cv2.inRange(imageHSV, min_HSV1, max_HSV1)
skinRegionHSV2 = cv2.inRange(imageHSV, min_HSV2, max_HSV2)
skinRegionHSV3 = cv2.inRange(imageHSV, min_HSV3, max_HSV3)

# menggabungkan gambar asli dengan mask skinRegionHSV
# mask berada di atas gambar asli
faceHSV1 = cv2.bitwise_and(image, image, mask = skinRegionHSV1)
faceHSV2 = cv2.bitwise_and(image, image, mask = skinRegionHSV2)
faceHSV3 = cv2.bitwise_and(image, image, mask = skinRegionHSV3)

# range ke-1
cv2.imshow("mask_HSV_1", skinRegionHSV1)
cv2.imshow("Face HSV_1", faceHSV1)
# range ke-1
cv2.imshow("mask_HSV_2", skinRegionHSV2)
cv2.imshow("Face HSV_2", faceHSV2)
# range ke-1
cv2.imshow("mask_HSV_3", skinRegionHSV3)
cv2.imshow("Face HSV_3", faceHSV3)

cv2.waitKey(0)
cv2.destroyAllWindows()

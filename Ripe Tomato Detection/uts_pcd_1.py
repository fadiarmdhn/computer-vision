import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tomato_2.jpg")

# konversi image dari RGB menjadi beberapa colorspace
imageHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", imageHSV)
imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
cv2.imshow("YCrCb", imageYCrCb)
imageLAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
cv2.imshow("LAB", imageLAB)
imageHLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
cv2.imshow("HLS", imageHLS)

# Histogram setiap channel pada colorspace
L,A,B=cv2.split(imageLAB)
cv2.imshow("L1_Channel",L) # For L Channel
cv2.imshow("A_Channel",A) # For A Channel (Here's what You need)
cv2.imshow("B_Channel",B) # For B Channel

histL = plt.hist(L.ravel(), 256, [0,256])
plt.show()
histA = plt.hist(A.ravel(), 256, [0,256])
plt.show()
histB = plt.hist(B.ravel(), 256, [0,256])
plt.show()

# menerapkan threshold ke gambar LAB untuk menentukan range atas dan bawah
# batas bawah lab1
min_LAB1 = np.array([30,150,54],np.uint8)
# batas atas lab1
max_LAB1 = np.array([200,255,200],np.uint8)
# batas bawah lab2
min_LAB2 = np.array([20,120,160],np.uint8)
# batas atas lab2
max_LAB2 = np.array([170,255,255],np.uint8)
# batas bawah lab3
min_LAB3 = np.array([10,132,100],np.uint8)
# batas atas lab3
max_LAB3 = np.array([255,255,255],np.uint8)

# membuat mask dengan
# mengembalikan nilai biner image menggunakan range bawah dan atas (min, max)
ripeMaskLAB1 = cv2.inRange(imageLAB,min_LAB1,max_LAB1)
ripeMaskLAB2 = cv2.inRange(imageLAB,min_LAB2,max_LAB2)
ripeMaskLAB3 = cv2.inRange(imageLAB,min_LAB3,max_LAB3)

# menggabungkan gambar asli dengan mask
# mask berada di atas gambar asli
tomatoLAB1 = cv2.bitwise_and(img, img, mask = ripeMaskLAB1)
tomatoLAB2 = cv2.bitwise_and(img, img, mask = ripeMaskLAB2)
tomatoLAB3 = cv2.bitwise_and(img, img, mask = ripeMaskLAB3)

# LAB 1
cv2.imshow("mask_LAB_1", ripeMaskLAB1)
cv2.imshow("tomato LAB_1", tomatoLAB1)
# LAB 2
cv2.imshow("mask_LAB_2", ripeMaskLAB2)
cv2.imshow("tomato LAB_2", tomatoLAB2)
# LAB 3
cv2.imshow("mask_LAB_3", ripeMaskLAB3)
cv2.imshow("tomato LAB_3", tomatoLAB3)

cv2.waitKey(0)
cv2.destroyAllWindows()
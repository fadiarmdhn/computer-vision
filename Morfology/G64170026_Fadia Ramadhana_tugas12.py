import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('tomato.jpg')

#lakukan konversi colorspace yang sesuai
imageLAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
cv2.imshow("LAB", imageLAB)

L,a,b = cv2.split(imageLAB)

fig, axes = plt.subplots(1, 3, figsize=(6, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(L, cmap="gray")
ax[0].set_title('L Channel')

ax[1].imshow(a, cmap="gray")
ax[1].set_title('a* Channel')

ax[2].imshow(b, cmap="gray")
ax[2].set_title('b* Channel')

#lakukan thresholding pada citra channel hasil konversi yang sesuai
# batas bawah a*
min_a = 136
# batas atas b*
max_a = 255
maskLAB = cv2.inRange(a, min_a, max_a)
cv2.imshow("Thresholding result", maskLAB)

def kernel(size):
    k = np.ones((size,size), np.uint8)
    return k

#lakukan proses dilasi
dilatation_3 = cv2.dilate(maskLAB, kernel(3), iterations=1)
dilatation_5 = cv2.dilate(maskLAB, kernel(5), iterations=1)
fig, axes = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(dilatation_3, cmap="gray")
ax[0].set_title('Dilatation with 3*3 kernel')

ax[1].imshow(dilatation_5, cmap="gray")
ax[1].set_title('Dilatation with 5*5 kernel')

#lakukan proses erosi
erotion_3 = cv2.erode(dilatation_3, kernel(3), iterations=1)
erotion_5 = cv2.erode(dilatation_3, kernel(5), iterations=1)
fig, axes = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(erotion_3, cmap="gray")
ax[0].set_title('Erotion with 3*3 kernel')

ax[1].imshow(erotion_5, cmap="gray")
ax[1].set_title('Erotion with 5*5 kernel')

#lakukan masking
masked = cv2.bitwise_and(img, img, mask = erotion_5)
cv2.imshow("masked", masked)
fig, axes = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap="gray")
ax[0].set_title('Input Image')

ax[1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB), cmap="gray")
ax[1].set_title('Masked')

plt.show()
cv2.waitKey()

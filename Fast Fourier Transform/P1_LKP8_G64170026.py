import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image
img = cv2.imread('leaf-spot-disease-control-in-michigan.jpg')
cv2.imshow("image original", img)
row,col,ch = img.shape
print(row, col, ch)

# Crop Image
imgcrop = img[:375,:600]

# Image Pre-Processing
img = cv2.GaussianBlur(imgcrop,(5,5),0)
cv2.imshow("Gaussian Blur 5*5", img)

# konversi image dari RGB menjadi beberapa colorspace
imageHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
imageLAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
imageHLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

plt.subplot(221), plt.imshow(imageHSV), plt.title("HSV")
plt.subplot(222), plt.imshow(imageYCrCb), plt.title("YCrCb")
plt.subplot(223), plt.imshow(imageLAB), plt.title("LAB")
plt.subplot(224), plt.imshow(imageHLS), plt.title("HLS")
plt.show()

# batas bawah YCrCb
min_ycrcb = np.array([0,135,0],np.uint8)
# batas atas YCrCb
max_ycrcb = np.array([255,255,255],np.uint8)

# membuat mask dengan
# mengembalikan nilai biner image menggunakan range bawah dan atas (min, max)
Mask = cv2.inRange(imageYCrCb, min_ycrcb, max_ycrcb)

# menggabungkan gambar asli dengan mask
# mask berada di atas gambar asli
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
segmentation = cv2.bitwise_and(img, img, mask = Mask)

plt.subplot(121), plt.imshow(Mask), plt.title("Mask")
plt.subplot(122), plt.imshow(segmentation), plt.title("Disease Segmentation")
plt.show()

segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)

plt.subplot(151), plt.imshow(segmentation), plt.title("Original Image")

# Fourier Transform
original = np.fft.fft2(segmentation)
plt.subplot(152), plt.imshow(np.log(1+np.abs(original)), cmap="gray"), plt.title("Spectrum")

center = np.fft.fftshift(original)
plt.subplot(153), plt.imshow(np.log(1+np.abs(center)), cmap="gray"), plt.title("Centered Spectrum")

inv_center = np.fft.ifftshift(center)
plt.subplot(154), plt.imshow(np.log(1+np.abs(inv_center)), cmap="gray"), plt.title("Decentralized")

processed_img = np.fft.ifft2(inv_center)
plt.subplot(155), plt.imshow(np.abs(processed_img), cmap="gray"), plt.title("Processed Image")

plt.show()

# Spectrum and Phase Angle Plot
plt.subplot(121), plt.imshow(np.log(np.abs(original)), "gray"), plt.title("Spectrum")

plt.subplot(122), plt.imshow(np.angle(original), "gray"), plt.title("Phase Angle")
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
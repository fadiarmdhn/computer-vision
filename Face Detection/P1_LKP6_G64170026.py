import numpy as np
import cv2

img = cv2.imread("FACE DETECTION.png")

def hsv(image):
    # mengakses ukuran citra
    row,col,ch = image.shape
    # membuat canvas baru berukuran matrix 3*3
    # dan semua elemennya berisi angka "0"
    canvas_1 = np.zeros((row,col,3), np.uint8)
    for i in range(0,row):
        for j in range(0,col):
            # mengakses nilai pixel RGB
            blue, green, red = image[i,j]
            # mengubah nilai pixel menjadi integer
            b = int(blue)
            g = int(green)
            r = int(red)

            # mengakses nilai pixel maksimal dari masing-masing channel
            maksimal = max(red, green, blue)
            # mengakses nilai pixel minimal dari masing-masing channel
            minimal = min(red, green, blue)

            # nilai V (Value)
            v = maksimal

            # nilai S (Saturation)
            if v != 0:
                s = (v-minimal)/v
            else:
                v = 0

            # nilai H (Hue)
            if v == minimal:
                h = 0
            elif v == r:
                h = (60*(g-b))/(v-minimal)
            elif v == g:
                h = 120+(60*(b-r))/(v-minimal)
            elif v == b:
                h = 240+(60*(r-g))/(v-minimal)

            # konversi nilai H, S, dan V ke tipe data tujuan
            # dalam hal ini adalah 8-bit images
            if h<0:
                h = h+360
            h = int(h/2)
            s = int(s*255)

            # memasukkan masing-masing nilai h, s, dan v ke dalam canvas_1
            canvas_1.itemset((i, j, 0), h)
            canvas_1.itemset((i, j, 1), s)
            canvas_1.itemset((i, j, 2), v)
    return canvas_1

final_hsv = hsv(img)
lib_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
(h,s,v) = cv2.split(lib_hsv)

cv2.imshow("HSV using library", lib_hsv)
cv2.imshow("HSV using function", final_hsv)

def YCrCb(image):
    # mengakses ukuran citra
    row,col,ch = image.shape
    # membuat canvas baru berukuran matrix 3*3
    # dan semua elemennya berisi angka "0"
    canvas_2 = np.zeros((row,col,3),np.uint8)
    # deklarasi nilai delta untuk 8-bit image
    delta = 128
    for i in range(0,row):
        for j in range(0,col):
            # mengakses nilai pixel RGB
            blue, green, red = image[i,j]
            # mengubah nilai pixel menjadi integer
            b = int(blue)
            g = int(green)
            r = int(red)

            # nilai Y (Luminance)
            Y = 0.299*r+0.587*g+0.114*b
            # nilai Cr (Crominance red)
            Cr = (r-Y)*0.713 + delta
            # nilai Cb (Criminance blue)
            Cb = (b-Y)*0.564 + delta

            # nilai R (red)
            R = Y + 1.403*(Cr-delta)
            # nilai G (green)
            G = Y - 0.714*(Cr-delta) - 0.344*(Cb-delta)
            # nilai B (blue)
            B = Y + 1.773*(Cb-delta)

            # memasukkan masing-masing nilai Y, Cr, dan Cb ke dalam canvas_2
            canvas_2.itemset((i, j, 0), Y)
            canvas_2.itemset((i, j, 1), Cr)
            canvas_2.itemset((i, j, 2), Cb)
    return canvas_2

final_YCrCb = YCrCb(img)
lib_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

cv2.imshow("YCrCb using library", lib_YCrCb)
cv2.imshow("YCrCb using function", final_YCrCb)

cv2.waitKey(0)
cv2.destroyAllWindows()
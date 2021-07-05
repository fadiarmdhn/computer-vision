import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Load image flower full
full = cv2.imread('flowers.tif')
# Convert BGR to RGB channels
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
cv2.imshow("flowers full",full)

# height, width, channels
fullshp = full.shape
print(fullshp)

# Load image flower template
template = cv2.imread('flower-template-resize.tif')
# Convert BGR to RGB channels
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
cv2.imshow("flowers template",template)

# height, width, channels
templateshp = template.shape
print(templateshp)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
    # Create a copy of image
    full_copy = full.copy()

    method = eval(m)

    # Template Matching
    result = cv2.matchTemplate(full, template, method)
    min_val, max_val, min_location, max_location = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_location
    else:
        top_left = max_location

    height, width, channels = template.shape
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(full_copy, top_left, bottom_right, (0, 0, 255), 10)

    # Plot and show the images
    plt.subplot(121)
    plt.imshow(result)
    plt.title('HEATMAP OF TEMPLATE MATCHING')

    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('DETECTION OF TEMPLATE')

    # Title with the method used
    plt.suptitle(m)

    plt.show()
    print('\n\n\n')

#------------------------------------------------------
# CONVOLUTION
full2 = cv2.imread('flowers.tif', 0)
template2 = cv2.imread('flower-template-resize.tif', 0)
full2shp = full2.shape
template2shp =template2.shape

k1 = ([-1/3, 0, 1/3],
      [-1/3, 0, 1/3],
      [-1/3, 0, 1/3])


k2 = ([-1/3, -1/3, -1/3],
      [0, 0, 0],
      [1/3, 1/3, 1/3])

def show(image, result):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    bottom_right = max_loc
    height, width = template2.shape
    top_left = (bottom_right[0] - width, bottom_right[1] - height)
    cv2.rectangle(image,top_left, bottom_right, (0, 0, 255), 10)

    # Plot and show the images
    plt.subplot(121)
    plt.imshow(result,cmap = 'gray')
    plt.title('HEATMAP OF TEMPLATE MATCHING')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(image,cmap = 'gray')
    plt.title('DETECTION OF TEMPLATE')
    plt.xticks([]), plt.yticks([])

    plt.suptitle('CONVOLUTION')
    plt.show()

# Get Gradient
Gfull2 = convolve2d(full2, k1) + convolve2d(full2, k2)
Gtemplate2 = convolve2d(template2, k1) + convolve2d(template2, k2)

# Flip template image along all axes to make
# the convolution behave like correlation
flip = np.rot90(Gtemplate2, 2)

# Perform the cross-correlation (by means of convolution
convolved = convolve2d(Gfull2, flip)
show(full2, convolved)


cv2.waitKey()
cv2.destroyAllWindows()


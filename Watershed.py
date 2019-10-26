import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Brightness, contrast control - recommend B = 0, C = 60

s = 128
def apply_brightness_contrast(input_img, brightness = 0, contrast = 60):
    input_img = cv2.resize(input_img, (s,s), 0, 0, cv2.INTER_AREA)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

font = cv2.FONT_HERSHEY_SIMPLEX
fcolor = (0,0,0)

blist = [0, -127, 127,   0,  0, 64] # list of brightness values
clist = [0,    0,   0, -64, 64, 64] # list of contrast values

out = np.zeros((s*2, s*3, 3), dtype = np.uint8)

for i, b in enumerate(blist):
    c = clist[i]
    print('b, c:  ', b,', ',c)
    row = s*int(i/3)
    col = s*(i%3)

    print('row, col:   ', row, ', ', col)

    out[row:row+s, col:col+s] = apply_brightness_contrast(img, b, c)
    msg = 'b %d' % b
    cv2.putText(out,msg,(col,row+s-22), font, .7, fcolor,1,cv2.LINE_AA)
    msg = 'c %d' % c
    cv2.putText(out,msg,(col,row+s-4), font, .7, fcolor,1,cv2.LINE_AA)
    cv2.putText(out, 'OpenCV',(260,30), font, 1.0, fcolor,2,cv2.LINE_AA)
    

# Read image, make gray scale, threshold to binary
img = cv.imread('mouse_brain-one_FOV.tif')
# Enhance
img = apply_brightness_contrast(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

plt.figure("Watershed Segmentation")
# plt.subplot(1, 6, 1)
# plt.imshow(img)
# plt.title('Original Image')
# plt.subplot(1, 6, 2)
# plt.imshow(thresh)
# plt.title('Binarized/threshold')

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=5)
# plt.subplot(1, 4, 1)
# plt.imshow(opening)
# plt.title('Morphological Opening')

# finding sure background area
sure_bg = cv.dilate(opening, kernel, iterations=1)
plt.subplot(2, 3, 1)
plt.imshow(sure_bg)
plt.title('Dilation:sure_bg')

# distance transform to find sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
print("dist_transform is: ", dist_transform.max())
ret, sure_fg = cv.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
plt.subplot(2, 3, 2)
plt.imshow(dist_transform)
plt.title('dist transform')
plt.subplot(2, 3, 3)
plt.imshow(sure_fg)
plt.title('thresholds: sure_fg')

# Finding unknown region
sure_fg = np.uint8(sure_fg)
dist_transform = np.uint8(dist_transform)
print(sure_fg.dtype)
print(sure_bg.dtype)
# unknown = cv.subtract(sure_bg, sure_fg)
unknown = cv.subtract(sure_bg, dist_transform)
plt.subplot(2, 3, 4)
plt.imshow(unknown)
plt.title('unknown')

# mark Labeling
ret, markers = cv.connectedComponents(dist_transform, labels=None, connectivity=8)
# Adding one to sure background, so it is 1 instead of 0
markers = markers + 1
# marking the unknown region as zero
markers[unknown == 255] = 0
markers = cv.watershed(img, markers)
plt.subplot(2, 3, 5)
plt.imshow(markers)
plt.title('markers')

img[markers == -1] = [255, 0, 0]
plt.subplot(2, 3, 6)
plt.imshow(img)
plt.title('Marked Image post segmentation')

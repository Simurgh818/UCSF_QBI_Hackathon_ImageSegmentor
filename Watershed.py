import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from queue import Queue


def get_nuclei_centroids(copy):
    im = copy.copy()

    shape_count = 0

    queue = Queue()

    shape_coords_all = []

    for i in range(len(im)):
        for j in range(len(im[0])):
            if im[i][j] == 0:  # if we find a black pixel, iterate count up
                shape_count += 1
                shape_coords = [(i, j)]
                im[i][j] = 1000  # mark the spot with invalid pixel so it won't be counted again
                queue.put((i, j))
                while not queue.empty():
                    cur_i, cur_j = queue.get()
                    if cur_i - 1 >= 0 and im[cur_i - 1][cur_j] == 0:
                        im[cur_i - 1][cur_j] = 1000
                        queue.put((cur_i - 1, cur_j))
                        shape_coords.append((cur_i - 1, cur_j))

                    if cur_i + 1 <= len(im) - 1 and im[cur_i + 1][cur_j] == 0:
                        im[cur_i + 1][cur_j] = 1000
                        queue.put((cur_i + 1, cur_j))
                        shape_coords.append((cur_i + 1, cur_j))

                    if cur_j - 1 >= 0 and im[cur_i][cur_j - 1] == 0:
                        im[cur_i][cur_j - 1] = 1000
                        queue.put((cur_i, cur_j - 1))
                        shape_coords.append((cur_i, cur_j - 1))

                    if cur_j + 1 <= len(im[0]) - 1 and im[cur_i][cur_j + 1] == 0:
                        im[cur_i][cur_j + 1] = 1000
                        queue.put((cur_i, cur_j + 1))
                        shape_coords.append((cur_i, cur_j + 1))

                x_sum, y_sum = 0, 0
                for coord in shape_coords:
                    x_sum += coord[0]
                    y_sum += coord[1]
                centroid = int(round(x_sum / len(shape_coords))), int(round(y_sum / len(shape_coords)))
                shape_coords_all.append(centroid)

    return shape_coords_all


print("Centroid function created")

# Read image, make gray scale, threshold to binary
img = cv.imread('mouse_brain-one_FOV.tif')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

plt.figure("Watershed Segmentation")
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original Image')

# noise removal
kernel = np.ones((5, 5), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)


# finding sure background area
sure_bg = cv.dilate(opening, kernel, iterations=1)
# inverting the dist. transform to get better sure_bg_v2
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
plt.subplot(2, 3, 2)
plt.imshow(sure_bg)
plt.title('sure_bg')

# distance transform to find sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
print("dist_transform is: ", dist_transform.max())
ret, sure_fg = cv.threshold(dist_transform, 0.15*dist_transform.max(), 255, 0)

# plt.subplot(2, 3, 2)
# plt.imshow(dist_transform)
# plt.title('dist transform')
plt.subplot(2, 3, 3)
plt.imshow(sure_fg)
plt.title('thresholds: sure_fg')

# Finding unknown region
sure_fg = np.uint8(sure_fg)
sure_bg = np.uint8(sure_bg)
dist_transform = np.int8(dist_transform)

unknown = cv.subtract(sure_bg, sure_fg)
unknown = np.uint8(unknown)
unknown_not = cv.bitwise_not(unknown)
plt.subplot(2, 3, 4)
plt.imshow(unknown_not)
plt.title('Nuclei segment boundaries')

# mark Labeling
ret, markers = cv.connectedComponents(sure_bg, labels=None, connectivity=8)
# Adding one to sure background, so it is 1 instead of 0
markers = markers + 1
# marking the unknown region as zero
markers[unknown == 255] = 0
# markers[unknown == 255] = -1
thresh_not = cv.bitwise_not(thresh)
centroids = get_nuclei_centroids(thresh)
print("The nuclei centroids are: ", centroids)
# markers[centroids] = 1

# x and y arrays
x, y = zip(*centroids)
print("x value is : ", x)
print("y value is :", y)

# Watershed function
img_watershed = cv.watershed(img, markers)
# img_watershed[unknown == 255] = -1
plt.subplot(2, 3, 5)
plt.imshow(img_watershed)
plt.title('img_watershed')

img[markers == -1] = [255, 0, 0]
img[unknown == 255] = [255, 0, 0]
plt.subplot(2, 3, 6)
plt.imshow(img)
plt.scatter(y, x, s=1, c='b')
plt.title('Marked Image post segmentation')
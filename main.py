import descriptor_generators as desc
import numpy as np
import time
import cv2
import glob

# where to look for images
images = glob.glob('INRIAPerson\96X160H96\Train\pos\*.png')     # image path and extension

# compute descriptors from image dataset, save in a file
desc.get_hog_descriptors(images, "descriptors_tp")

# read in the descriptors
descriptors_loaded = np.load("descriptors_tp.npy")
#print len(descriptors_loaded)


sample_image = cv2.imread(images[0])
cv2.imshow("win", sample_image)
cv2.waitKey()

width, height, depth = sample_image.shape

for cols in range(0, width-20):
    for rows in range(0, height-20):
        img_copy = sample_image.copy()
        cv2.rectangle(img_copy, (rows, cols), (rows+20, cols+20), (255, 255, 0))
        key = cv2.waitKey(1)
        cv2.imshow("win", img_copy)

cv2.waitKey(10)


import numpy as np
import cv2
from skimage.feature import hog

def get_hog_descriptors(images, result_filename):
    img_cnt = 0                 # init image count to zero
    descriptor_vector = []      # empty descriptor list
    print "Processing dataset images..."
    for filename in images:
        # read image, convert to grayscale
        current_image = cv2.imread(filename)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        # compute the descriptor, append to the descriptor vector
        descriptor = hog(current_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        descriptor_vector.append(descriptor)
        # formatted output information
        print "processing image: ", img_cnt, "out of: ", images.__len__(),
        print "     {:.3f}".format(img_cnt*100.0/images.__len__()), "%\r",
        img_cnt += 1

    # save the descriptor vector to a numpy file
    print "Saving the descriptors to file..."
    np.save(result_filename, descriptor_vector)
    print "Done!"

#coding : utf-8

import numpy as np
import cv2
from mycvlib import *

point_set = read_point_set("point_set.txt")

trans_matrix = create_trans_matrix(point_set)
img = cv2.imread("homo2.jpg", 0)
save_image(img, "homo_source.png")
img_height = img.shape[0]
img_width = img.shape[1]
print('width = ' + str(img_width) + 'px')
print('height = ' + str(img_height) + 'px')
save_img = np.zeros((img_height, img_width))
homography_translation(img, save_img, trans_matrix)
# b = np.array([213, 28, 1])
# transpoint = np.dot(trans_matrix, b)
# print(trans_matrix)
# print(transpoint)
# print("in" + str(source_img[213][28]))
# print("out" + str(save_img[63][0]))
save_image(save_img, "homo.png")

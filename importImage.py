#coding : utf-8

import numpy as np
import cv2
from mycvlib import *

# img = cv2.imread("spike.jpg", 0)
# save_image(img, "median_source.png")

img1 = cv2.imread("hdr1.jpg", 0)
img2 = cv2.imread("hdr2.jpg", 0)
img3 = cv2.imread("hdr3.jpg", 0)
#
# save_image(img1, "hdr1.png")
# save_image(img2, "hdr2.png")
# save_image(img3, "hdr3.png")

img_height = img1.shape[0]
img_width = img1.shape[1]
print('width = ' + str(img_width) + 'px')
print('height = ' + str(img_height) + 'px')
#処理画像フレーム生成
save_img = np.zeros((img_height, img_width))
HDR(img1, img2, img3, save_img)
#処理画像の保存
save_image(save_img, "HDRnon.png")

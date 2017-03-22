#coding : utf-8

import numpy as np
import cv2
from mycvlib import *


img = cv2.imread("bi_source.png", 0)
save_image(img, "bi_source2.png")
img_height = img.shape[0]
img_width = img.shape[1]
print('width = ' + str(img_width) + 'px')
print('height = ' + str(img_height) + 'px')
#処理画像フレーム生成
save_img = np.zeros((img_height, img_width))


bilateral_filtering(img, save_img, 2, 4, 11)
print(save_img)
#処理画像の保存
save_image(save_img, "bi.png")

#coding : utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_image(input_img_name):
    img = cv2.imread(input_img_name, 0)
    return img


def save_image(save_img, file_name):
    cv2.imwrite(file_name, save_img)


def show_image(show_img):
    cv2.imshow("result_image", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#ガンマ変換
def gamma_translation(gamma, input_img, output_img):
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]
    for i in range(img_height):
        for j in range(img_width):
            tmp_pixel = 255 * (input_img[i][j] / 255) ** (1 / gamma)
            output_img.itemset((i, j), tmp_pixel)

#ヒストグラム平坦化
def equalize_histgram(input_img, output_img):

    PIXEL_VALUE_MAX = 256
    img_hist = [0 for i in range(PIXEL_VALUE_MAX)]
    img_add_hist = [0 for i in range(PIXEL_VALUE_MAX)]
    hist_table = [0 for i in range(PIXEL_VALUE_MAX)]
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]
    hist_max = 0
    add_hist_max = img_height * img_width

    #ヒストグラムの取得
    for i in range(img_height):
        for j in range(img_width):
            tmp = input_img[i][j]
            img_hist[tmp] += 1

    #ヒスグラム最大値
    hist_max = max(img_hist)

    #累積ヒストグラムの取得
    for i in range(PIXEL_VALUE_MAX):
        if i == 0:
            img_add_hist[i] = img_hist[i]
        else:
            img_add_hist[i] = (img_add_hist[i - 1] + img_hist[i])

    #ヒストグラム、累積ヒストグラムの正規化
    for i in range(PIXEL_VALUE_MAX):
        img_hist[i] = img_hist[i] / hist_max
        img_add_hist[i] = img_add_hist[i] / add_hist_max

    #変換テーブルの作成
    for i in range(PIXEL_VALUE_MAX):
        tmp_value = img_add_hist[i] * 255
        hist_table[i] = tmp_value

    for i in range(img_height):
        for j in range(img_width):
            pix_value = input_img[i][j]
            output_img[i][j] = hist_table[pix_value]


#空間フィルタリングにおける行列演算
def image_filtering(filter_matrix, input_img, output_img):
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]

    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    output_img[i][j] += filter_matrix[k + 1][l + 1] * input_img[i + k][j + l]
            int(output_img[i][j])


#平滑化
def smoothed_filtering(input_img, output_img):
    smoothed_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    image_filtering(smoothed_filter, input_img, output_img)


#微分
def diff_filtering(input_img, output_img):
    diff_filter = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    image_filtering(diff_filter, input_img, output_img)
#鮮鋭化
def sharpened_filtering(input_img, output_img):
    sharpened_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharpened_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_filtering(sharpened_filter, input_img, output_img)


#エッジ検出
def edge_detection_filtering(input_img, output_img):
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]
    yoko_result = np.zeros((img_height, img_width))
    tate_result = np.zeros((img_height, img_width))
    edge_yoko_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edge_tate_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    image_filtering(edge_yoko_filter, input_img, yoko_result)
    image_filtering(edge_tate_filter, input_img, tate_result)
    for i in range(img_height):
        for j in range(img_width):
            tmp = np.sqrt(yoko_result[i][j] ** 2 + tate_result[i][j] ** 2)
            if tmp > 255:
                output_img[i][j] = 255
            elif tmp < 0:
                output_img[i][j] = 0
            else:
                output_img[i][j] = tmp


#メディアンフィルタ
def median_filtering(n, m, input_img, output_img):
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]

    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            output_img[i][j] = median(n, m, input_img, i, j)


#中央値の算出
def median(n, m, input_img, i, j):
    median_value = 0
    count = 0
    median_list = np.zeros(n * m)
    for k in range(-1, 2):
        for l in range(-1, 2):
            median_list[count] = input_img[i + k][ j + l]
            count += 1
    median_value = np.median(median_list)
    return int(median_value)


#バイラテラルフィルタ
def bilateral_filtering(input_img, output_img, s1, s2, W):
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]
    work = np.zeros((img_height, img_width))
    for i in range(W, img_height - W):
        for j in range(W, img_width - W):
            weighten_pix = 0
            sum_weight = 0
            for k in range(-W, W + 1):
                for l in range(-W, W + 1):
                    im1 = input_img[i][j]
                    im2 = input_img[i + k][j + l]
                    w2_tmp = im1 - im2
                    w2_tmp = (w2_tmp) * (w2_tmp) / -(2 * s2 ** 2)
                    weight = np.exp(-(k ** 2 + l ** 2) / (2 * s1 ** 2)) * np.exp(w2_tmp)
                    sum_weight += weight
                    weighten_pix += im2 * weight
            work[i][j] = weighten_pix / sum_weight
            output_img[i][j] = int(work[i][j])

#ハーフトーン(ディザー法)
def dither_halftone(input_img, output_img):
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]
    dmatrix = np.array([[0, 8, 2, 10],[12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
    block_height = img_height // 4
    block_width = img_width // 4
    for i in range(0, block_height):
        for j in range(0, block_width):
            for k in range(0, 4):
                for l in range(0, 4):
                    yp = i * 4 + k
                    xp = j * 4 + l
                    if input_img[xp][yp] <= (dmatrix[k][l] * 16 + 8):
                        output_img[xp][yp] = 0
                    else:
                        output_img[xp][yp] = 255


#ハーフトーン(誤差拡散法)
def error_diffusion_halftone(input_img, output_img):
    img_height = input_img.shape[0]
    img_width = input_img.shape[1]
    work = np.zeros((img_height, img_width), np.uint8)
    for i in range(0, img_height):
        for j in range(0, img_width -0):
            work[i][j] = input_img[i][j]

    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            if work[i][j] <= 127:
                error = work[i][j] - 0
                work[i][j] = 0
            else:
                error = work[i][j] - 255
                work[i][j] = 255
            work[i + 1][j] += (5 / 16) * error
            work[i + 1][j + 1] += (3 / 16) * error
            work[i][j + 1] += (5 / 16) * error
            work[i - 1][j + 1] += (3 / 16) * error

    for i in range(img_height):
        for j in range(img_width):
            output_img[i][j] = 255

    for i in range(0, img_height - 1):
        for j in range(1, img_width - 1):
            if work[i][j] > 255:
                output_img[i][j] = 255
            elif work[i][j] < 0:
                output_img[i][j] = 0
            else:
                output_img[i][j] = int(work[i][j])


#HDR画像合成
def HDR(input_img1, input_img2, input_img3, output_img):
    img_height = output_img.shape[0]
    img_width = output_img.shape[1]
    work = np.zeros((img_height, img_width))
    for i in range(img_height):
        for j in range(img_width):
            #単純合成
            work[i][j] = (input_img1[i][j] * 1.2 + input_img2[i][j] * 1.75 + input_img3[i][j] * 0.05) / 3
            if work[i][j] > 255:
                output_img[i][j] = 255
            elif work[i][j] < 0:
                output_img[i][j] = 0
            else:
                output_img[i][j] = int(work[i][j])


#射影変換
def homography_translation(input_img, output_img, trans_matrix):
    img_height = output_img.shape[0]
    img_width = output_img.shape[1]
    for u in range(img_height):
        for v in range(img_width):
            point = point_translation(u, v, trans_matrix)
            point_x = int(point[0] + 0.5)
            point_y = int(point[1] + 0.5)
            if point_x >=  img_height or point_y >= img_width:
                output_img[u][v] = 0
            else:
                output_img[u][v] = input_img[int(point_x + 0.5)][int(point_y + 0.5)]


#u, vは変換後の座標、x, yは変換前の座標
def point_translation(u, v, trans_matrix):
    point = np.dot(trans_matrix, np.array([u, v, 1]))
    x = point[0]
    y = point[1]
    return (x, y)



#指定した座標セットから変換行列を求める
def create_trans_matrix(point_set):
    x = point_set[0]
    y = point_set[1]
    u = point_set[2]
    v = point_set[3]
    #A_mat * xp = b
    # trans_matrix = np.zeros((3,3))
    A_mat = np.zeros((8,8))
    b = np.zeros(8)
    xp = np.zeros(8)
    for i in range(0, 7, 2):
        j = i + 1
        k = int(i / 2)
        print("i is " + str(i))
        print("j is " + str(j))
        print("k is " + str(k))
        A_mat[i][0] = x[k]
        A_mat[i][1] = y[k]
        A_mat[i][2] = 1
        A_mat[i][3] = 0
        A_mat[i][4] = 0
        A_mat[i][5] = 0
        A_mat[i][6] = x[k] * u[k]
        A_mat[i][7] = y[k] * u[k]

        b[i] = u[k]

        A_mat[j][0] = 0
        A_mat[j][1] = 0
        A_mat[j][2] = 0
        A_mat[j][3] = x[k]
        A_mat[j][4] = y[k]
        A_mat[j][5] = 1
        A_mat[j][6] = x[k] * v[k]
        A_mat[j][7] = y[k] * v[k]

        b[j] = v[k]
    xp = np.linalg.solve(A_mat, b)
    trans_matrix = np.array([[xp[0], xp[1], xp[2]], [xp[3], xp[4], xp[5]], [xp[6], xp[7], 1]])
    # trans_matrix[0][0] = x_mat[0]
    # trans_matrix[0][1] = x_mat[1]
    # trans_matrix[0][2] = x_mat[2]
    # trans_matrix[1][0] = x_mat[3]
    # trans_matrix[1][1] = x_mat[4]
    # trans_matrix[1][2] = x_mat[5]
    # trans_matrix[2][0] = x_mat[6]
    # trans_matrix[2][1] = x_mat[7]
    # trans_matrix[2][2] = 1
    trans_matrix = np.linalg.inv(trans_matrix)
    print(A_mat)
    return trans_matrix


def read_point_set(file):
    count = 0
    x = np.zeros(4)
    y = np.zeros(4)
    u = np.zeros(4)
    v = np.zeros(4)
    for line in open(file, "r"):
        data = line.split()
        if count < 4:
            x[count] = data[0]
            y[count] = data[1]
        else:
            u[count - 4] = data[0]
            v[count - 4] = data[1]
        count += 1
    return (x, y, u, v)

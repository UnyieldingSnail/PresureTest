# -*- coding: utf-8 -*-
# feimengjuan
# 利用python实现多种方法来实现图像识别

import cv2
import numpy as np
from functools import reduce
from skimage import morphology


# from matplotlib import pyplot as plt


# 最简单的以灰度直方图作为相似比较的实现
def classify_gray_hist(image1, image2, size=(256, 256)):
    # 先计算直方图
    # 几个参数必须用方括号括起来
    # 这里直接用灰度图计算直方图，所以是使用第一个通道，
    # 也可以进行通道分离后，得到多个通道的直方图
    # bins 取为16
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 可以比较下直方图
    # plt.plot(range(256), hist1, 'r')
    # plt.plot(range(256), hist2, 'b')
    # plt.show()
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


# 计算单通道的直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


# 通过得到每个通道的直方图来计算相似度
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# 平均哈希算法计算
def classify_aHash(image1, image2):
    image1 = cv2.resize(image1, (8, 8))
    image2 = cv2.resize(image2, (8, 8))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1, hash2)


def classify_pHash(image1, image2):
    image1 = cv2.resize(image1, (32, 32))
    image2 = cv2.resize(image2, (32, 32))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1, hash2)


# 输入灰度图，返回hash
def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


def wordArea_filter(img):
    # 提取图片中的黑色字
    res_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    LowerBlack = np.array([0, 0, 0])
    UpperBlack = np.array([188, 255, 100])
    black_mask = cv2.inRange(res_hsv, LowerBlack, UpperBlack)
    # words_img = cv2.bitwise_and(img, img, mask=black_mask)
    cv2.imshow("Image", black_mask)
    cv2.waitKey(0)
    return black_mask


def wordArea_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    mean = np.mean(gray[gray < mean])
    mean = np.mean(gray[gray < mean])
    _, img = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("gray2", img)
    cv2.waitKey(0)
    return img


def imgray_match_templates(gray1, gray2):
    # gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

    # 检测是否空白
    _, img = cv2.threshold(gray1, 100, 255, cv2.THRESH_BINARY_INV)
    (x, y) = np.nonzero(img)
    if x.size == 0:
        return -1

    # 过滤黑色字
    mean1 = np.mean(gray1)
    mean1 = np.mean(gray1[gray1 < mean1])
    mean2 = np.mean(gray2)
    mean2 = np.mean(gray2[gray2 < mean2])
    mean2 = np.mean(gray2[gray2 < mean2])
    # mean = (mean1 + mean2) / 2
    # print(mean1, mean2)
    _, img1 = cv2.threshold(gray1, mean1, 255, cv2.THRESH_BINARY_INV)
    _, img2 = cv2.threshold(gray2, mean2, 255, cv2.THRESH_BINARY_INV)

    # 截取模版字的最小外接矩形
    (x, y) = np.nonzero(img2)
    x_min = np.min(x)
    x_max = np.max(x) + 1
    y_min = np.min(y)
    y_max = np.max(y) + 1
    img2 = img2[x_min:x_max, y_min:y_max]
    # print(x_min, x_max, y_min, y_max)
    # print(img2)
    # cv2.imshow("gray1", img1)
    # cv2.imshow("gray2", img2)
    # cv2.waitKey(0)

    # 模版匹配
    match = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)

    # 获取最大匹配值
    _, confidence, _, _ = cv2.minMaxLoc(match)
    e = np.exp(confidence * 4)
    score = e / (e + 1) * 100
    # print(confidence, score)
    return score


def matchModels(src1, src2):
    gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    mean1 = np.mean(gray1)
    mean1 = np.mean(gray1[gray1 < mean1])
    mean2 = np.mean(gray2)
    mean2 = np.mean(gray2[gray2 < mean2])
    # print(mean1, mean2)
    _, img1 = cv2.threshold(gray1, 175, 1, cv2.THRESH_BINARY_INV)
    _, img2 = cv2.threshold(gray2, 175, 1, cv2.THRESH_BINARY_INV)
    # print(img1)
    # img2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("gray1", img1)
    # cv2.imwrite("./1.jpg", img1)
    # cv2.imshow("gray2", img2)
    # cv2.imwrite("./2.jpg", img2)
    # cv2.waitKey(0)

    # 骨架提取
    img1 = morphology.skeletonize(img1)
    img2 = morphology.skeletonize(img2)
    # print(img1)
    # cv2.imshow("gray1", img1)
    # cv2.imwrite("./1.jpg", img1)
    # cv2.imshow("gray2", img2)
    # cv2.imwrite("./2.jpg", img2)
    # cv2.waitKey(0)
    # 获取非零像素的坐标
    (y1, x1) = np.nonzero(img1)
    (y2, x2) = np.nonzero(img2)
    cnt1 = map(lambda x, y: [[x, y]], x1, y1)
    cnt2 = map(lambda x, y: [[x, y]], x2, y2)
    cnt1 = np.array(list(cnt1))
    cnt2 = np.array(list(cnt2))
    # _, cnt1_arr, _ = cv2.findContours(img1, 3, 2)
    # _, cnt2_arr, _ = cv2.findContours(img2, 3, 2)
    # cnt1 = reduce(lambda x, y: np.r_[x, y], cnt1_arr)
    # cnt2 = reduce(lambda x, y: np.r_[x, y], cnt2_arr)

    # cv2.drawContours(src1, cnt1, -1, (0, 0, 255), 1)
    # cv2.drawContours(src2, cnt2, -1, (0, 0, 255), 1)
    # cv2.imshow("Image", src1)
    # cv2.waitKey(0)
    # cv2.imshow("Image", src2)
    # cv2.waitKey(0)

    # print(cnt1)
    # print(cnt2)
    print(cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I2, 0.0))

    # degree = my_ssim(img1, img2)
    # (degree, diff) = compare_ssim(img1, img2, full=True)
    # degree = classify_gray_hist(img1, img2)
    # degree = classify_hist_with_split(img1,img2)
    # degree = classify_aHash(img1,img2)


if __name__ == '__main__':
    # for i in range(8):
    path1 = "./img/0-0.jpg"
    path2 = "./img/0-3.jpg"
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    imgray_match_templates(img1, img2)

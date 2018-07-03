# -*- coding: utf-8 -*-
from sklearn import cluster

import cv2
import sys
import math
import numpy as np


img_path = sys.argv[1]
img = cv2.imread(img_path)
# img=cv2.blur(img,(1,1))

gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean1 = np.mean(gray1)
mean1 = np.mean(gray1[gray1 < mean1])
# print(mean1, mean2)
_, img1 = cv2.threshold(gray1, 125, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("0", img1)
cv2.waitKey(0)

img = cv2.GaussianBlur(img, (3, 3), 0)
imgray = cv2.Canny(img, 50, 100, 3)  # Canny边缘检测，参数可更改
cv2.imwrite("./imgray.jpg", imgray)
cv2.imshow("0", imgray)
cv2.waitKey(0)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
centers = []
for cnt in contours:
    if len(cnt) > 50:
        S1 = cv2.contourArea(cnt)
        ell = cv2.fitEllipse(cnt)
        S2 = math.pi * ell[1][0] * ell[1][1]
        if (S1 / S2) > 0.2:  # 面积比例，可以更改，根据数据集。。。
            img = cv2.ellipse(img, ell, (0, 255, 0), 2)
            print(str(S1) + "    " + str(S2) + "   " + str(ell))
            centers.append(ell[0])
print(centers)
labels = cluster.KMeans(n_clusters=4, random_state=170).fit_predict(centers)
print(labels)
cv2.imshow("0", img)
cv2.imwrite("./center.jpg", img)
cv2.waitKey(0)

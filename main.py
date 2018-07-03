# -*- coding: utf-8 -*-
import sys
import os
import uuid
import cv2
import logging.config
import pandas as pd
from util import *
from sklearn import cluster

def main(_path):
    conf_path = os.path.join(os.path.dirname(__file__), 'conf')
    logging_config_path = os.path.join(conf_path, 'logging.conf')
    logging.config.fileConfig(logging_config_path)
    logger = logging.getLogger("root")

    img_path = _path

    logger.info("img_path：%s" % img_path)

    # 读取图像，支持 bmp、jpg、png、tiff 等常用格式
    img_src = cv2.imread(img_path)
    logger.info("图片加载完成：%s" % img_path)
    height_src, width_src, dim_src = img_src.shape
    width = BlockConf().resize[0]
    height = int(height_src * width / width_src)

    # 图片中心点
    center_point = Point(width / 2, height / 2)

    # 调整图像大小
    logger.info("开始调整图片大小")
    size = (width, height)
    img_resize = cv2.resize(img_src, size, interpolation=cv2.INTER_CUBIC)
    logger.info("图片大小调整为：(%s, %s)" % (width, height))

    # 转化为HSV图像
    logger.info("提取图片中绿色区域")
    HSV = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)

    # 提取绿色区域
    LowerGreen = np.array([34, 15, 46])
    UpperGreen = np.array([107, 255, 255])
    mask = cv2.inRange(HSV, LowerGreen, UpperGreen)
    img = img_resize.copy()
    img[mask != 0] = [255, 255, 255]
    # words_img = cv2.bitwise_xor(img, cv2.bitwise_not(img), mask=mask)
    filtered_area = mask
    logger.info("绿色区域提取完毕")
    # cv2.imwrite("filtered_area.jpg", img)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    # 寻找轮廓
    binary, contours, hierarchy = cv2.findContours(filtered_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 显示所有轮廓
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    # cv2.imwrite("empty.jpg", img)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    # 最小边界面积
    min_area = width * height * 0.5
    contour = findAppositeContour(contours, min_area)
    logger.info("轮廓提取完毕:%s" % len(contour))

    # 创建空白图像并用于绘制所找到的轮廓
    if len(contour) == 0:
        logger.error("未找到有效轮廓")
        exit(1)
    empty_img = np.zeros(img.shape, np.uint8)
    cv2.drawContours(empty_img, [contour], -1, (255, 255, 255), 2)

    # 显示有效轮廓
    # cv2.imwrite("effective_contour.jpg", empty_img)
    # cv2.imshow("Image", empty_img)
    # cv2.waitKey(0)

    # 霍夫直线检测
    logger.info("执行霍夫变换直线检测")
    gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLines(gray, 1, np.pi / 180, 160)

    # 提取为为二维
    lines = lines[:, 0, :]

    # 需要检查是否为四条直线
    logger.info("检测出%s条直线" % len(lines))
    logger.info("检测出的直线：%s" % lines)
    # lines = lines[lines[:, 1] < np.pi / 2]

    # 处理直线数据用于聚类 train_lines[dis, x, y]
    lines[:, 1][lines[:, 0] < 0] = lines[:, 1][lines[:, 0] < 0] + np.pi
    lines[:, 0] = np.abs(lines[:, 0])
    train_lines = np.c_[lines[:, 0] / height, np.cos(lines[:, 1]), np.sin(lines[:, 1])]

    # 对直线数据做聚类处理
    labels = cluster.KMeans(n_clusters=4, random_state=170).fit_predict(train_lines)
    logger.info("直线的类别：%s" % labels)

    # 求每一类直线的均值直线
    lines = [pd.DataFrame(lines[labels == i]).agg({0: 'mean', 1: middle_degree}) for i in range(4)]
    logger.info("聚类处理后的直线：%s" % lines)

    # 对rho, theta表示的直线进行处理封装
    my_points = parseLines(lines, center_point, img)
    # cv2.imwrite("line_img.jpg", img)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    # 选出矩形的四个顶点
    logger.info("计算出的交点为：%s" % my_points)
    my_points.sort(key=lambda _: _.dis)
    vertexes = my_points[:4]
    logger.info("选出的矩形的四个顶点为：%s" % my_points)
    orderVertexes(vertexes)
    logger.info("按照左上右上左下右下排序的顶点为：：%s" % my_points)

    # 透视校正
    std_shape = BlockConf().resize
    src = np.float32([[v.x, v.y] for v in vertexes])
    dst = np.float32([[0, 0], [std_shape[0], 0], [0, std_shape[1]], [std_shape[0], std_shape[1]]])
    M = cv2.getPerspectiveTransform(src, dst)

    # 用作汉字切割个打分
    res = cv2.warpPerspective(img, M, std_shape)

    # 用作标记分数和不及格汉字
    img_score_mark = cv2.warpPerspective(img_resize, M, std_shape)

    # 切割
    words = cutWords(res)
    (row, col) = BlockConf().shape
    for i in range(row):
        for j in range(col):
            cv2.imwrite("img/%s-%s.jpg" % (i, j), words[i, j])

    # 标记不及格的字
    mean_score = markWorse(img_score_mark, words)

    # 保存结果图片
    dir_path = os.path.join(os.path.dirname(__file__), 'result')
    file_name = "%s.jpg" % uuid.uuid1()
    result_path = os.path.join(dir_path, file_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    cv2.imwrite(result_path, img_score_mark)
    print(os.path.join("result", file_name + ":" + str(mean_score)))



if __name__ == '__main__':
    main()

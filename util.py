# -*- coding: utf-8 -*-
from entity import *
from sysconf import BlockConf
from similary import *

blockConf = BlockConf()
shape = blockConf.shape
row = blockConf.rowRatio
col = blockConf.colRatio
words_area = np.zeros((shape[0], shape[1], 4), np.int32)
(width, height) = blockConf.resize
length = int(171 * width / 2000)
for i in range(shape[0]):
    for j in range(shape[1]):
        words_area[i, j] = (
            int(row[i] * height), int(row[i] * height) + length, int(col[j] * width),
            int(col[j] * width) + length)


def findAppositeContour(contours, min_area):
    area = float("inf")
    contour = []
    for c in contours:
        temp_area = cv2.contourArea(c)
        if min_area < temp_area < area:
            area = temp_area
            contour = c
    return contour


# 左上 右上 左下 右下
def orderVertexes(vertexes):
    vertexes.sort(key=lambda _: _.y)
    if vertexes[0].x > vertexes[1].x:
        vertex = vertexes[1]
        vertexes[1] = vertexes[0]
        vertexes[0] = vertex
    if vertexes[2].x > vertexes[3].x:
        vertex = vertexes[3]
        vertexes[3] = vertexes[2]
        vertexes[2] = vertex


def cutWords(img):
    words = np.zeros((shape[0], shape[1], length, length, 3), np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            (x1, x2, y1, y2) = words_area[i, j]
            # print(x1, x2, y1, y2)
            words[i, j] = img[x1:x2, y1:y2]
    return words


def GetLinePara(line):
    line.a = line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x * line.p2.y - line.p2.x * line.p1.y


def GetCrossPoint(l1, l2):
    GetLinePara(l1)
    GetLinePara(l2)
    d = l1.a * l2.b - l2.a * l1.b
    if d == 0:
        return None
    else:
        p = Point()
        p.x = (l1.b * l2.c - l2.b * l1.c) * 1.0 / d
        p.y = (l1.c * l2.a - l2.c * l1.a) * 1.0 / d
        return p


def middle_degree(ids):
    less = ids[ids < np.pi]
    greater = ids[ids >= np.pi]
    if len(less) == 0:
        return sum(greater) / len(greater)
    elif len(greater) == 0:
        return sum(less) / len(less)
    else:
        greater_mean = sum(greater) / len(greater)
        less_mean = sum(less) / len(less)
        if abs(less_mean - greater_mean) > np.pi:
            return (less_mean + greater_mean + 2 * np.pi) / 2 % (2 * np.pi)
        else:
            return (less_mean + greater_mean) / 2


def parseLines(lines, center_point, img=None):
    my_lines = []
    my_points = []
    # 对rho, theta表示的直线进行处理封装
    for rho, theta in lines[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        pStart = Point(x1, y1)
        pEnd = Point(x2, y2)
        line = Line(pStart, pEnd)
        # 计算当前直线与已知直线的交点并存入数组
        if len(my_lines) != 0:
            for l in my_lines:
                p = GetCrossPoint(l, line)
                if p:
                    p.dis = pow(center_point.x - p.x, 2) + pow(center_point.y - p.y, 2)
                    my_points.append(p)
        # 将直线添加到数组中
        my_lines.append(line)
        # 在图片中画出直线
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return my_points


def evaluate(std, temps):
    scores = []
    std = cv2.cvtColor(std, cv2.COLOR_BGR2GRAY)
    for temp in temps:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        score = imgray_match_templates(std, temp)
        scores.append(int(score))
    return scores


def markWorse(img, words):
    radius = length / 2
    row = words.shape[0]
    col = words.shape[1]
    score_sum = 0.0
    word_sum = 0
    for r in range(row):
        # print("-------------------------------------------")
        scores = evaluate(words[r, 0], words[r, 1:col])
        for c in range(len(scores)):
            if scores[c] < 0:
                continue
            word_sum += 1
            (y, _, x, _) = words_area[r, c + 1]
            if scores[c] < 0.6:
                # 标记
                # (y, _, x, _) = words_area[r, c+1]
                # print(x, y)
                # print(r, c)
                x += radius
                y += radius
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 255), 1)
            score_sum += scores[c]
            cv2.putText(img, str(scores[c]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # 画出分数
    mean_score = int(score_sum / word_sum)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(mean_score), (int(width * 0.075), int(height * 0.075)), font, 2, (0, 0, 255), 2)
    return mean_score


if __name__ == "__main__":
    p1 = Point(0, 1)
    p2 = Point(1, 0)
    line1 = Line(p1, p2)

    p3 = Point(0, 2)
    p4 = Point(2, 1)
    line2 = Line(p3, p4)
    Pc = GetCrossPoint(line1, line2)
    if Pc:
        print("Cross point:", Pc.x, Pc.y)

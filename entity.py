# -*- coding: utf-8 -*-
class Point(object):
    x = 0
    y = 0
    dis = float("inf")

    # 定义构造方法
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return "(%s, %s, %s)" % (self.x, self.y, self.dis)

    __repr__ = __str__


class Line(object):
    # a=0
    # b=0
    # c=0
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

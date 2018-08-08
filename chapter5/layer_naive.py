# -*- coding: utf-8 -*-
# @Time         : 2018-08-05 20:11
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : layer_naive.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    @staticmethod
    def forward(x, y):
        return x + y

    @staticmethod
    def backward(dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

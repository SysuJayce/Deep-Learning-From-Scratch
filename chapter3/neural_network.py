# -*- coding: utf-8 -*-
# @Time         : 2018-08-01 17:42
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : neural_network.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np


def sigmoid(x):
    y = 1.0 / (1 + np.exp(-x))
    return y


def identity_func(x):
    return x


def softmax(x):
    c = np.max(x)
    return np.exp(x-c) / np.sum(np.exp(x-c))


def init_network():
    network = dict()
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 不能用W1.T * x
    # 因为1. W1, x都只是数组，并不是矩阵，应该使用np.dot
    # 2.如果使用*，那就是按元素相乘，而不是矩阵乘法
    # 按元素相乘的话就会使用到numpy的广播
    # 使用np.dot的时候可以把一维数组的维数看成只有1个数字，不是按照矩阵那样看
    # 也就是说，np.dot(a, b)，如果a是一维的，有n个元素，b是二维的，有n*m个元素
    # 可以直接用np.dot(a, b)，而如果要调换顺序，需要np.dot(b.T, a)
    # 前者是  n   n*m       后者是  m*n   n，这样相邻的维度都一样就可以约减
    a1 = np.dot(W1.T, x) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(W3.T, z2) + b3
    y = identity_func(a3)

    return y


def main():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)


if __name__ == '__main__':
    main()

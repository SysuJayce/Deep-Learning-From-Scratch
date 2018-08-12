# -*- coding: utf-8 -*-
# @Time         : 2018-08-11 22:47
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : visualize_filter.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt

from chapter7.simple_convnet import SimpleConvNet


def filter_show(filters, nx=8, margin=3):
    """
    根据滤波器参数绘制成一张灰度图。一个滤波器就绘制一个子图
    :param filters: 滤波器参数
    :param nx: 一行有几个子图(列数)
    :param margin: 预留边缘
    :return:
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))  # 行数

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05,
                        wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap='gray', interpolation='nearest')
    plt.show()


def main():
    network = SimpleConvNet()
    # 随机进行初始化后的权重
    filter_show(network.params['W1'])

    # 学习后的权重
    network.load_params()
    filter_show(network.params['W1'])


if __name__ == '__main__':
    main()

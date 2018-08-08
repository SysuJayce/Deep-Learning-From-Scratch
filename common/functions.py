# -*- coding: utf-8 -*-
# @Time         : 2018-08-02 20:17
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : functions.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np


def identify_function(x):
    """
    恒等函数：一般用作回归任务时的输出层激活函数
    :param x:
    :return:
    """
    return x


def step_function(x):
    """
    阶跃函数：早期单层感知机使用的激活函数
    :param x:
    :return:
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    """
    阶跃函数的升级版，具有可连续求导、非线性的良好性质。但是缺点在于容易梯度消失
    :param x:
    :return:
    """
    return 1.0 / (1 + np.exp(-x))


def relu(x):
    """
    求导简单，作用类似于一个开关，输入<=0时输出为0，导数为0
    输入>0时输出不变，导数为1
    :param x:
    :return:
    """
    return np.maximum(x, 0)


def softmax(x):
    """
    可以将输入规范化成概率，在分类时用处很大。一般是在训练过程和交叉熵一起用。
    在softmax后输入到交叉熵损失函数中计算损失，然后反向传播更新参数。
    但是在训练完成后的推理阶段一般可以不用，计算输出层的argmax即可得出分类结果
    :param x:
    :return:
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 为防止指数计算时溢出，减去最大值
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(true_label, prediction):
    """
    均方误差。一般用作回归任务中的损失函数。前面乘0.5可以在求导时约掉系数
    :param true_label:
    :param prediction:
    :return:
    """
    return 0.5 * np.sum((true_label-prediction)**2)


def cross_entropy_error(true_label, prediction):
    """
    交叉熵误差损失函数：当使用均方误差作为损失函数的时候，
    反向传播过程会出现激活函数的求导项，对于像sigmoid这样容易梯度消失的激活函数
    如果使用均方误差就可能遇到梯度消失的问题。使用交叉熵误差的话在求梯度的时候
    不会出现激活函数的导数项，因此在一定程度上比均方误差性能好。
    :param true_label:
    :param prediction:
    :return:
    """
    delta = 1e-7  # 设置一个极小值来防止计算ln(0)时溢出下界

    # 维度为1代表向量(array数组)，维度为2代表矩阵
    # 这里将向量转换成矩阵形式
    if prediction.ndim == 1:
        true_label = true_label.reshape(1, true_label.size)
        prediction = prediction.reshape(1, prediction.size)

    # 如果预测值和真实值的大小一样，说明真实值是以one-hot编码的
    # 因此将真实值转换成one-hot编码的索引
    # 这时true_label表示预测值array中用于计算熵大小的下标
    # 即 -sum(ln(pred[true_label]))
    if true_label.size == prediction.size:
        true_label = true_label.argmax(axis=1)

    batch_size = prediction.shape[0]  # 如果是以batch进行训练就求平均误差
    return -np.sum(np.log(prediction[np.arange(batch_size), true_label] +
                          delta)) / batch_size


def sigmoid_grad(x):
    """
    sigmoid函数的导数
    :param x:
    :return:
    """
    return (1.0 - sigmoid(x)) * sigmoid(x)

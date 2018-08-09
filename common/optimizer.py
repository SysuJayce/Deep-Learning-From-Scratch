# -*- coding: utf-8 -*-
# @Time         : 2018-08-09 16:03
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : optimizer.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np


class SGD:
    """
    随机梯度下降优化器
    学习率保持不变
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    仿照物理动力学中的动量来设计的神经网络学习优化器
    每个待优化参数都维护一个速度，而对应的梯度则作为力。注意速度和力都有方向，
    力决定速度的改变，而需要维护一个momentum变量作为阻力。
    因此不只有梯度和学习率可以影响参数值，阻力momentum也可以影响参数值。

    当梯度很小、但是一直朝着同一个方向的时候，SGD对参数的改变很小很慢，
    而Momentum则可以加速这个参数的学习。
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.v = None  # 变量v维护参数的速度
        self.lr = lr
        self.momentum = momentum  # 阻力系数

    def update(self, params, grads):
        # 初始化每个参数的速度，以字典形式保存在v中
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 更新参数：经过阻力降速后的速度，然后施加力(梯度)改变速度
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """
    AdaGrad通过将学习率随着训练的进行逐步减小。
    在开始的时候学习率较大，然后随着训练的进行，学习率逐渐变小

    AdaGrad为每个参数适当调整学习率，每个参数记录过去自己的所有梯度的平方和h，
    学习率随h更新为：lr/sqrt(h)，  h越来越大，则学习率越来越小，最终变为0
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] ** 2
            # 在分母开根号的时候加上一个微量，防止出现分母为0的情况
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)


class RMSProp:
    """
    RMSProp是对AdaGrad的改进。
    因为AdaGrad随着训练的进行学习率最后会变为0，不再更新参数。

    为了改善这个问题，RMSProp通过逐渐遗忘过去的梯度，然后补充新梯度。
    这样可以避免h过大导致最后学习率为0，也增加了新梯度的影响力，更符合实际。
    """
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate  # 遗忘过去梯度的比例
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate  # 遗忘部分过去的梯度
            # 新梯度补充遗忘掉的比例，也就是说遗忘多少就从新梯度中补充多少
            self.h[key] += (1 - self.decay_rate) * grads[key] ** 2
            # 在更新参数的方式和AdaGrad一样
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)


class Adam:
    """
    Adam融合了Momentum和AdaGrad的思想
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        # 更新学习率。由于beta1比beta2要小，随着iter的增加，分母越来越大，
        # 分子增加的速率比分母慢，因此self.lr的系数是逐渐变小的，于是学习率变小
        lr_t = (self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) /
                (1.0 - self.beta1 ** self.iter))

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

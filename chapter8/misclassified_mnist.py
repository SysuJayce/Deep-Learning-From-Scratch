# -*- coding: utf-8 -*-
# @Time         : 2018-08-13 16:32
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : misclassified_mnist.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import matplotlib.pyplot as plt

from chapter8.deep_convnet import DeepConvNet
from common.mnist import load_mnist


def draw(classified_ids, test_x, test_label):
    """
    将测试结果和真实标签进行对比，然后输出分类错误的测试样例
    :param classified_ids: 预测结果
    :param test_x: 测试集特征
    :param test_label: 测试集真实标签
    :return:
    """
    max_view = 20  # 输出前20个错误样例
    current_view = 1

    # 绘制画布
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05,
                        wspace=0.05)

    mis_pairs = {}  # 记录预测错误的pair
    for i, val in enumerate(classified_ids == test_label):
        # 如果预测错误
        if not val:
            ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
            ax.imshow(test_x[i].reshape(28, 28), cmap='gray',
                      interpolation='nearest')
            mis_pairs[current_view] = (test_label[i], classified_ids[i])

            current_view += 1
            if current_view > max_view:
                break

    print("======= misclassified result =======")
    print("{view index: (label, inference), ...}")
    print(mis_pairs)

    plt.show()


def get_predict_result(network, test_x, test_label):
    batch_size = 100
    classified_ids = []

    print("calculating test accuracy ... ")
    # 如果真实标签是one-hot编码，先提取成一维数组：一行代表一个真实值
    if test_label.ndim != 1:
        true_label = np.argmax(test_label, axis=1)

    correct_cnt = 0.0
    # 书上原本代码没有处理剩余的这部分数据。
    # 在这里加上iters这个变量，用于控制循环次数
    if test_x.shape[0] % batch_size:
        iters = int(test_x.shape[0] / batch_size) + 1
    else:
        iters = int(test_x.shape[0] / batch_size)

    for i in range(iters):
        # 获取一个batch的数据和对应的真实标签
        temp_x = test_x[i * batch_size: (i + 1) * batch_size]
        temp_true_label = test_label[i * batch_size: (i + 1) * batch_size]
        # 预测这个batch的数据的输出
        y = network.predict(temp_x)
        y = np.argmax(y, axis=1)
        classified_ids.append(y)  # 将每次预测的结果保存起来
        # 统计每个batch的预测正确数
        correct_cnt += np.sum(y == temp_true_label)

    acc = correct_cnt / test_x.shape[0]  # 计算准确率
    print("test accuracy:" + str(acc))

    return classified_ids


def main():
    _, (test_x, test_label) = load_mnist(flatten=False)
    # sampled = 1000
    # test_x = test_x[:sampled]
    # test_label = test_label[:sampled]

    network = DeepConvNet()  # 构造神经网络
    network.load_params()  # 加载已经训练好的参数

    # 对测试集进行预测，获得每个测试样例的预测结果
    classified_ids = get_predict_result(network, test_x, test_label)
    classified_ids = np.array(classified_ids).flatten()

    # 输出预测错误的图像
    draw(classified_ids, test_x, test_label)


if __name__ == '__main__':
    main()

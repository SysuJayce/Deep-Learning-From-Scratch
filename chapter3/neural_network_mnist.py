# -*- coding: utf-8 -*-
# @Time         : 2018-08-02 20:01
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : neural_network_mnist.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
import pickle

from common.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_test_data():
    """
    获取测试集数据
    :return:
    """
    _, (test_X, test_y) = load_mnist(normalize=True,
                                     flatten=True,
                                     one_hot_label=False)
    return test_X, test_y


def init_network():
    """
    使用已经训练好的持久化pickle文件来直接初始化网络参数
    :return:
    """
    with open('./data/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 由于现在使用了batch，因此输入的x就是100x784，而W1是784x50，所以不用转置
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return np.argmax(y, axis=1)


def main():
    batch_size = 100
    accuracy_cnt = 0.0

    test_X, test_y = get_test_data()
    network = init_network()

    for i in range(0, len(test_X), batch_size):
        pred = predict(network, test_X[i: i+batch_size])
        accuracy_cnt += np.sum(pred == test_y[i: i+batch_size])  # 统计正确个数

    accuracy = accuracy_cnt / len(test_X)
    print("Accuracy: %f" % accuracy)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
# @Time         : 2018-08-02 15:12
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : mnist.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import os
import os.path
import urllib.request
import gzip
import pickle
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'  # MNIST下载地址
key_file = {  # 训练集和测试集的下载地址(feature和label)
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

img_size = 784  # 每张图片的大小，当需要返回多维数组的时候用到
# 数据集保存目录
dataset_dir = '../chapter3/data'
save_path = os.path.join(dataset_dir, 'mnist.pkl')  # pickle文件的保存地址


def _download(file_name):
    """
    下载MNIST的file_name文件到datasetdir目录
    :param file_name: 待下载的文件名
    :return:
    """
    file_path = os.path.join(dataset_dir, file_name)  # 下载文件的保存地址
    if os.path.exists(file_path):  # 不重复下载
        print("%s already exist." % file_name)
        return

    print("Downloading %s ..." % file_name)
    # 使用urllib.request.urlretrieve()下载文件
    urllib.request.urlretrieve(url_base+file_name, file_path)
    print("Done")


def download_mnst():
    """
    下载全部mnist的四个文件
    :return:
    """
    for v in key_file.values():
        _download(v)


def _load_label(file_name):
    """
    读取下载好的label文件中的label
    :param file_name:
    :return:
    """
    file_path = os.path.join(dataset_dir, file_name)
    print("Converting %s to NumPy Array..." % file_name)
    with gzip.open(file_path, 'rb') as f:
        # 读取label，格式为np.uint8，偏移8的原因可能是前面8个字符保存了别的信息
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name):
    """
    读取下载好的img文件中的数据
    :param file_name:
    :return:
    """
    file_path = os.path.join(dataset_dir, file_name)
    print("Converting %s to NumPy Array..." % file_name)
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, img_size)  # 默认将每个样本展平成一维数组
    print("Done")

    return data


def _convert_numpy():
    """
    利用前面定义好的两个辅助函数：_load_img和_load_label读取MNIST数据并保存在
    dataset这个字典中
    :return: dataset字典
    """
    dataset = dict()
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_mnist():
    """
    初始化程序，包括下载MNIST、转化成NumPy数组、使用pickle持久化到硬盘
    :return:
    """
    download_mnst()
    dataset = _convert_numpy()
    print("Creating pickle file...")
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    """
    使用one-hot编码label
    :param X:
    :return:
    """
    # 先生成一个全零的矩阵，每个样本一行，每个样本有10列
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        # 对每一个样本(每一行)将对应的数组中的其中一个0变成1
        # 使用枚举可以保证每个样本都有且只有一个1
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    加载MNIST数据集
    :param normalize: 是否将灰度值归一化
    :param flatten: 是否将每个样本的像素矩阵展平
    :param one_hot_label: 是否使用独热编码label
    :return: train_img, train_label, test_img, test_label，四个NumPy的多维数组
    """
    if not os.path.exists(save_path):  # 如果未找到持久化的pickle文件就重新下载
        init_mnist()

    with open(save_path, 'rb') as f:  # 读取pickle文件
        dataset = pickle.load(f)

    if normalize:  # 将灰度值归一化：每个灰度值除以255
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:  # 将train和test数据集的label用独热编码
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    # 默认是展平每一个样本的像素矩阵
    # 如果需要返回矩阵，reshape成-1,1,28,28
    # -1代表不指定有多少个这样的样本，根据实际数据量自动确定
    # 1 表示每个样本都是单通道
    # 28,28表示矩阵是28x28的。
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape((-1, 1, 28, 28))

    return (dataset['train_img'], dataset['train_label']),\
           (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()

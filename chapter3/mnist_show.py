# -*- coding: utf-8 -*-
# @Time         : 2018-08-02 16:13
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : mnist_show.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import numpy as np
from PIL import Image
from chapter3.mnist import load_mnist


def img_show(img_):
    pil_img = Image.fromarray(np.uint8(img_))
    pil_img.show()


(train_X, train_y), (test_X, test_y) = load_mnist(normalize=False,
                                                  flatten=True,
                                                  one_hot_label=False)

img = train_X[0]
label = train_y[0]
print(label)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
img_show(img)

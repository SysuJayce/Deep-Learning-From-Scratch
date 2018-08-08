# -*- coding: utf-8 -*-
# @Time         : 2018-08-05 20:16
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : buy_apple.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from chapter5.layer_naive import MulLayer


def main():
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    total_price = mul_tax_layer.forward(apple_price, tax)

    # backward
    dtotal_price = 1
    dapple_price, dtax = mul_tax_layer.backward(dtotal_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print("forward:\ntotal price: %d\n" % total_price)
    print("backward:\ndApple:", dapple)
    print("dApple_num:", int(dapple_num))
    print("dTax:", dtax)


if __name__ == '__main__':
    main()

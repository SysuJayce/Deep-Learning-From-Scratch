# -*- coding: utf-8 -*-
# @Time         : 2018-08-05 20:38
# @Author       : Jayce Wong
# @ProjectName  : Deep_Learning_From_Scratch
# @FileName     : buy_apple_orange.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from chapter5.layer_naive import MulLayer, AddLayer


def main():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    # layer
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    apple_orange_price = add_apple_orange_layer.forward(apple_price,
                                                        orange_price)
    total_price = mul_tax_layer.forward(apple_orange_price, tax)

    # backward
    dtotal_price = 1
    dapple_orange_price, dtax = mul_tax_layer.backward(dtotal_price)
    dapple_price, dorange_price =\
        add_apple_orange_layer.backward(dapple_orange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)

    print("forward:\ntotal price: %d\n" % total_price)
    print("backward:\ndApple:", dapple)
    print("dApple_num:", int(dapple_num))
    print("dOrange:", dorange)
    print("dOrange_num:", int(dorange_num))
    print("dTax:", dtax)


if __name__ == '__main__':
    main()

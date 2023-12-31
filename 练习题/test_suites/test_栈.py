#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/31 13:37
# @Author  : MinisterYU
# @File    : test_栈.py

from 练习题.栈.下一个更大的元素 import Solution as S1

s1 = S1()


def test_dailyTemperatures():
    res = s1.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73])
    print(res)
    assert res == [1, 1, 4, 2, 1, 1, 0, 0]

def test_next_greater_element_reversed():
    pass
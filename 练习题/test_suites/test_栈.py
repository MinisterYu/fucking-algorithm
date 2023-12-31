#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/31 13:37
# @Author  : MinisterYU
# @File    : test_栈.py
import logging
logging.basicConfig(level=logging.DEBUG)

from typing import List
from 练习题.栈.下一个更大的元素 import Solution as S1

s1 = S1()


def test_dailyTemperatures():
    res = s1.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73])
    print(res)
    assert res == [1, 1, 4, 2, 1, 1, 0, 0]


def next_greater_element_reversed(nums1: List[int], nums2: List[int]) -> List[int]:
    # https://leetcode.cn/problems/next-greater-element-i/
    # nums1 是nums2 的子集
    stack = []
    res = []
    mapper = {}
    for i in range(len(nums2)):
        while stack and nums2[stack[-1]] < nums2[i]:
            index = stack.pop()
            mapper[nums2[index]] = nums2[i]
        stack.append(i)

    while stack:
        mapper[nums2[stack.pop()]] = -1

    for i in range(len(nums1)):
        res.append(mapper[nums1[i]])
    return res


def test_next_greater_element_reversed():
    assert next_greater_element_reversed([4, 1, 2], [1, 3, 4, 2]) == [-1, 3, -1]

def originalDigits(s):
    from collections import OrderedDict, Counter
    mapping = OrderedDict({
        "zero": 0,
        "two": 2,
        "four": 4,
        "six": 6,
        "eight": 8,
        "one": 1,
        "three": 3,
        "five": 5,
        "seven": 7,
        "nine": 9
    })
    digits = []

    counter = Counter(s)
    for i in range(len(mapping)):
        word = mapping.keys()[i]
        digit = mapping.values()[i]
        count_word = counter.get(word)
        if count_word > 0:
            for char in word:
                counter[char] -= count_word
            digits.extend([digit] * count_word)

    # 按升序返回原始的数字
    digits.sort()
    return ''.join(str(i) for i in digits)

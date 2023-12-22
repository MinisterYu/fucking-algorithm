#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 10:53
# @Author  : MinisterYU
# @File    : 第一个缺失的正整数.py

def firstMissingPositive(nums):
    n = len(nums)

    # 将所有非正整数置为 n+1
    # 遍历数组 nums，将所有非正整数（小于等于0）都置为一个特殊的值，例如数组的长度加一（n + 1）。这是因为我们只关心正整数，而不关心非正整数。
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1
    print(nums)
    # 根据数组中的正整数将对应位置的数置为负数
    # 再次遍历数组 nums，对于每个正整数 num，我们将数组中索引为 num - 1 的位置的数置为负数。这样，我们可以通过正负号来标记某个正整数是否出现过。
    for i in range(n):
        num = abs(nums[i])
        if num <= n:
            nums[num - 1] = -abs(nums[num - 1])

    print(nums)
    # 找到第一个正数的位置，即为缺失的最小正整数
    # 最后，再次遍历数组 nums，找到第一个正数的位置，即为缺失的最小正整数。
    for i in range(n):
        if nums[i] > 0:
            return i + 1

    # 如果数组中都是正整数，则缺失的最小正整数为 n+1
    return n + 1


# 示例用法
nums = [3, 4, -1, 1]
result = firstMissingPositive(nums)
print(result)  # 输出 2

from collections import  defaultdict

d = defaultdict(list)
a = 'bdac'

print(a)
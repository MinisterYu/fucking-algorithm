#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 19:12
# @Author  : MinisterYU
# @File    : __init__.py.py


# TODO 最长连续序列  [100, 4, 200, 1, 3, 2]
def longestConsecutive(nums):
    num_set = set(nums)  # 创建一个哈希集合，用于快速查找数字
    max_len = 0  # 最长序列的长度

    for num in nums:
        if num - 1 not in num_set:  # 判断当前数字是否是序列的起点
            curr_num = num  # 当前连续序列的数字
            curr_len = 1  # 当前连续序列的长度

            while curr_num + 1 in num_set:  # 向后遍历连续序列
                curr_num += 1
                curr_len += 1

            max_len = max(max_len, curr_len)  # 更新最长序列的长度

    return max_len

print(longestConsecutive([100, 4, 200, 1, 3, 2]))

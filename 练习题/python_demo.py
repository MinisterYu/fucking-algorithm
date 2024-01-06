#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 23:19
# @Author  : MinisterYU
# @File    : python_demo.py
import collections
import itertools
import operator
import functools
import heapq

#
# # accumulate 前缀和
# data = [1, 2, 3, 4, 5, 6, 7]
# res = [0] + list(itertools.accumulate(data, operator.add))
# print(res)
#
# # functools.reduce 迭代计算
# res = functools.reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])
# print(res)
#
# dict_ = collections.defaultdict(lambda: 100)
# dict_[1] = 2
# print(dict_)
# print(dict_[2])
#
# heap = []
# heapq.heappush(heap, [1, 'a'])
# heapq.heappush(heap, [1, 'b'])
# heapq.heappush(heap, [2, 'a'])
# heapq.heappush(heap, [2, 'a'])
# print(heap)

# nums = [1, 4, 2, 5, 3]
# n = len(nums)
# for i in range(n):
#     print(n - i + 1)
#     for j in range(1, n - i + 1, 2):
#         print(nums[i: i + j])
from typing import List
from itertools import accumulate


#
#
# def leftRightDifference(nums: List[int]) -> List[int]:
#     left_prefix = [0] + list(accumulate(nums[:-1]))
#     right_prefix = list(accumulate(nums[::-1][:-1]))[::-1] + [0]
#     print(left_prefix)
#     print(right_prefix)
#
#
# leftRightDifference([10, 4, 8, 3])
#
# nums = [-5, 1, 5, 0, -7]
# s = [0] * (len(nums) + 1)
# for i in range(len(nums)):
#     s[i + 1] = s[i] + nums[i]
# print(s)

def numberOfPoints(nums: List[List[int]]) -> int:
    # 找到二维数组中所有终点的最大值
    max_end = max(end for _, end in nums)
    # 创建差分数组，长度为最大终点值加2
    diff = [0] * (max_end + 2)

    # 遍历二维数组中的每个起点和终点
    for start, end in nums:
        # 在起点位置加1，表示车辆从起点开始覆盖了一个整数点
        diff[start] += 1
        # 在终点的下一个位置减1，表示车辆到达终点后不再覆盖该整数点
        diff[end + 1] -= 1

    # 计算差分数组的累积和
    cumulative_sum = list(accumulate(diff))
    print(cumulative_sum)

    # 统计覆盖次数大于0的位置，即被车辆任意部分覆盖的整数点
    # result = sum(s > 0 for s in list(cumulative_sum))
    result = 0
    for i in cumulative_sum:
        print(i)
        if i > 0:
            result += 1

    print(result)


def removeDuplicates(s: str) -> str:
    '''
    输入："abbaca"
    输出："ca"
    '''
    stack = []
    for char in s:
        while stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)

    return ''.join(stack)


removeDuplicates('abbaca')

# s = '123456789'
# s = list(s)
# for i in range(len(s) - 1, 0, -1):
#     print(s[i - 1], s[i])

from itertools import pairwise
a = pairwise('000111000111')
print(list(a))

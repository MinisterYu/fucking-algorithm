#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 08:55
# @Author  : MinisterYU
# @File    : 一些遍历方法.py
import itertools

nums = [i for i in range(1, 10)]
n = len(nums)


def k_window_traverse_reverse(k=3):
    res = []
    for i in range(1, n + 1):
        i_res = []
        for j in range(1, min(i, k) + 1):
            i_res.append(nums[i - j])
        res.append(i_res)
    print(res)


def k_window_traverse(k=3):
    res = []
    for i in range(1, n + 1):
        i_res = []
        for j in range(min(i, k), 0, -1):  # 右边界取 K 和 I 的最小值，保障窗口大小不超过K
            i_res.append(nums[i - j])
        res.append(tuple(i_res))
    print(res)


print('按 k 大小的窗口滑动遍历')
# k_window_traverse_reverse()
k_window_traverse()
k_window_traverse(k=4)


def steps_traverse(start=0, k=2):
    res = []
    for i in range(n):
        if i * k > n:  # 边界超过数组，退出
            break
        if i * k + 1 > start:  # 右边界大于左边界，开始加数据
            i_res = []
            for j in range(start, i * k + 1, k):
                i_res.append(nums[j])
            res.append(i_res)
            # res.append(tuple(nums[start: i * k + 1:k]))
    print(res)


print('按K个跳过取值')
steps_traverse(0, 2)
steps_traverse(1, 2)
steps_traverse(2, 3)


def k_pairwise_traverse(k=2):
    # 123 , 234, 345, 456
    res = []
    for i in range(n):
        if i + k > n:
            break
        i_res = []
        for j in range(i, i + k):
            i_res.append(nums[j])
        res.append(i_res)
        # res.append(tuple(nums[i: i + k]))
    print(res)


print('按K个pairwise 取值')
print(list(itertools.pairwise(nums)))
k_pairwise_traverse(k=2)
k_pairwise_traverse(k=3)

# 二维数组
print('二维数组')
nums2 = [[i + 1 + n * 3 for i in range(3)] for n in range(3)]
for i in nums2:
    print(i)
m, n = len(nums2), len(nums2[0])


def upright_traverse(nums2):
    m, n = len(nums2), len(nums2[0])
    for i in range(m):
        for j in range(i):
            if i == j:
                continue
            nums2[i][j], nums2[j][i] = nums2[j][i], nums2[i][j]
    for i in nums2:
        print(i)


#
# print('二维数组竖过来')
# upright_traverse(nums2)


def clockwise_traverse(nums2):
    # upright_traverse(nums2)
    # print('---')
    m, n = len(nums2), len(nums2[0])

    mid = n // 2
    for i in range(mid):
        for j in range(m):
            nums2[j][i], nums2[j][n - mid] = nums2[j][n - mid], nums2[j][i]
    for i in nums2:
        print(i)


nums2 = [[i + 1 + n * 3 for i in range(3)] for n in range(3)]
print('二维数顺时针旋转')
clockwise_traverse(nums2)

nums2 = [[i + 1 + n * 3 for i in range(3)] for n in range(3)]

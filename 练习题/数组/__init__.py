#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 19:12
# @Author  : MinisterYU
# @File    : __init__.py.py

# TODO 冒泡
def bubble_sort(nums):
    n = len(nums)
    for x in range(n - 1):
        # 每次遍历将最大的元素冒泡到末尾
        for y in range(n - 1 - x):
            if nums[y] > nums[y + 1]:
                nums[y], nums[y + 1] = nums[y + 1], nums[y]

    return nums


def bubble_sort2(nums):
    n = len(nums)
    swap = True
    while swap:
        swap = False
        for i in range(1, n):
            if nums[i - 1] > nums[i]:
                nums[i], nums[i - 1] = nums[i - 1], nums[i]
                swap = True
    return nums


# TODO 归并
def merge_sort(nums):
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return _merged(left, right)


def _merged(left, right):
    res = []
    x, y = 0, 0
    while x < len(left) and y < len(right):
        if left[x] <= right[y]:
            res.append(left[x])
            x += 1
        else:
            res.append(right[y])
            y += 1
    res.extend(left[x:])
    res.extend(right[y:])
    return res


# TODO 快排
def quick_sort(nums):
    if len(nums) <= 1:
        return nums

    pivot = nums[len(nums) // 2]
    left = [x for x in nums if x < pivot]
    right = [x for x in nums if x > pivot]
    middle = [x for x in nums if x == pivot]

    return quick_sort(left) + middle + quick_sort(right)


# TODO 选择排序
def selection_sort(nums):
    n = len(nums)

    for x in range(n - 1):
        min_idx = x

        # 找到未排序部分中的最小元素的索引
        for y in range(x + 1, n):
            if nums[y] < nums[min_idx]:
                min_idx = y

        # 将最小元素与当前位置交换
        nums[x], nums[min_idx] = nums[min_idx], nums[x]

    return nums


# TODO 双指针排序
def sqaure_sort(nums):
    res = [0] * len(nums)
    left = 0
    right = len(nums) - 1
    x = len(nums) - 1
    while left <= right:
        if nums[left] <= nums[right]:
            res[x] = nums[right]
            right -= 1
        else:
            res[x] = nums[left]
            left += 1
        x -= 1

    return res


nums = [100, 4, 200, 1, 3, 2]
print(bubble_sort(nums))
print(bubble_sort2(nums))


# print(merge_sort(nums))
# print(quick_sort(nums))
# print(selection_sort(nums))
# print(sqaure_sort(nums))


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


# TODO  旋转数组  12345  -> 45123
def rotate(nums, k):
    n = len(nums)
    move = n - k % n
    '''
    1、切片
    nums[:] = nums[move:] + nums[:move]

    2、拼接切片
    nums = nums + nums
    nums[:] = nums[move :move + n]

    3、旋转赋值
    res = [0] * n
    for i in range(n):
        res[ (i + k) % n ] = nums[i]
    nums[:] = res

   '''


# todo 杨辉三角
def yh_trigle(numRows):
    triangle = [[1]]

    for _ in range(1, numRows):
        row = [1] + [a + b for a, b in zip(triangle[-1], triangle[-1][1:])] + [1]
        triangle.append(row)

    return triangle



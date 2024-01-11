#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 11:17
# @Author  : MinisterYU
# @File    : 哈希.py


def findMissingNumber(nums):
    # 给定一个长度为 n 的数组，其中的值是从 0 到 n 的范围内，且只有一个数字缺失，我们可以使用原地哈希的方法来找出缺失的数字。

    '''
    在这个示例中，我们首先遍历数组 nums，将出现过的数字进行标记。对于每个数字 nums[i]，
    我们将其对应的索引 index 处的数字取负数，表示该数字已经出现过。
    然后，我们再次遍历数组，找到第一个未被标记的数字，即为缺失的数字。
    如果没有找到未被标记的数字，则说明缺失的是 n。
    通过这种原地哈希的方式，我们可以在不使用额外空间的情况下找出缺失的数字。
    '''
    n = len(nums)

    # 标记出现过的数字
    for i in range(n):
        index = abs(nums[i])
        nums[index] = -abs(nums[index])

    # 找到未被标记的数字
    for i in range(n):
        if nums[i] > 0:
            return i

    # 如果没有未被标记的数字，则缺失的是 n
    return n

def findMissingPositive(nums):
    # 找出未排序的整数数组中没有出现的最小的正整数，可以使用原地哈希的方法。
    '''
    我们首先将所有非正整数置为 n+1，这样它们不会影响到最小的正整数的查找。
    然后，我们遍历数组，将出现过的正整数进行标记。
    对于每个正整数 num，我们将其对应的索引 num - 1 处的数字取负数，表示该正整数已经出现过。
    最后，我们再次遍历数组，找到第一个未被标记的正整数，即为没有出现的最小的正整数。
    如果都被标记了，则说明没有出现的最小的正整数是 n+1。
    通过这种原地哈希的方式，我们可以在不使用额外空间的情况下找出未排序的整数数组中没有出现的最小的正整数。
    '''
    n = len(nums)

    # 将所有非正整数置为 n+1
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1

    # 标记出现过的正整数
    for i in range(n):
        num = abs(nums[i])
        if num <= n:
            nums[num - 1] = -abs(nums[num - 1])

    # 找到第一个未被标记的正整数
    for i in range(n):
        if nums[i] > 0:
            return i + 1

    # 如果都被标记了，则返回 n+1
    return n + 1
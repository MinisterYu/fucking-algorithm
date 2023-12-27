#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 10:24
# @Author  : MinisterYU
# @File    : 下一个排列.py
from typing import List


def nextPermutation(nums: List[int]) -> None:
    """
    从右到左遍历数组，找到第一个相邻的元素对 (i, i+1)，满足 nums[i] < nums[i+1]。

    如果不存在这样的元素对，则说明数组已经是最大排列，直接将数组反转即可得到最小排列。

    在找到的元素对 (i, i+1) 中，从右到左找到第一个大于 nums[i] 的元素 nums[j]，并将其与 nums[i] 交换位置。

    将位置 i+1 及其右侧的元素进行反转，以得到下一个排列。
    """
    i = len(nums) - 2  # 从数组尾倒数第二个开始遍历， 找到第一组前一个数小于后一个数的
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i == -1:  # 如果没找到，则现在数组非递增，直接反转返回
        nums = nums[::-1]
        # nums.reverse()
        return

    j = len(nums) - 1
    while j > i and nums[i] >= nums[j]:  # 从右往左，找到第一个 第一个大于 nums[i] 的元素 nums[j]
        j -= 1

    nums[i], nums[j] = nums[j], nums[i]

    # 反转 i + 1后面的数字
    left, right = i + 1, len(nums) - 1
    while left <= right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
    # ----------------------
    n = len(nums)

    # 从数组尾倒数第二个开始遍历，找到第一组前一个数小于后一个数的
    for i in range(n - 2, -1, -1):
        if nums[i] < nums[i + 1]:
            break
    else:
        # 如果没找到，则现在数组非递增，直接反转返回
        nums.reverse()
        return

    # 从右往左，找到第一个大于 nums[i] 的元素 nums[j]
    for j in range(n - 1, i, -1):
        if nums[i] < nums[j]:
            break

    nums[i], nums[j] = nums[j], nums[i]

    # 反转 i + 1 后面的数字
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1

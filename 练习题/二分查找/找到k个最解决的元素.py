#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/23 23:45
# @Author  : MinisterYU
# @File    : 找到k个最解决的元素.py
from typing import List


class Solution:
    '''
    https://leetcode.cn/problems/find-k-closest-elements/description/?envType=study-plan-v2&envId=2024-spring-sprint-100
    给定一个 排序好 的数组 arr ，两个整数 k 和 x ，从数组中找到最靠近 x（两数之差最小）的 k 个数。返回的结果必须要是按升序排好的。
    整数 a 比整数 b 更接近 x 需要满足：

    |a - x| < |b - x| 或者
    |a - x| == |b - x| 且 a < b

    示例 1：
    输入：arr = [1,2,3,4,5], k = 4, x = 3
    输出：[1,2,3,4]
    '''

    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:

        left, right = 0, len(arr) - k

        while left < right:
            mid = left + (right - left) // 2
            # 如果 x - arr[mid] > arr[mid + k] - x，则说明最靠近 x 的 k 个数在 mid 的右侧。因此，我们将左指针 left 更新为 mid + 1。
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            # 如果 x - arr[mid] < arr[mid + k] - x，则说明最靠近 x 的 k 个数在 mid 的左侧或者 mid 就是其中之一。因此，我们将右指针 right 更新为 mid。
            else:
                right = mid

        return arr[left: left + k]

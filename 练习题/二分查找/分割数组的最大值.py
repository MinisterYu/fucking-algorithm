#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/23 22:59
# @Author  : MinisterYU
# @File    : 分割数组的最大值.py
from typing import List

class Solution:
    '''
    https://leetcode.cn/problems/split-array-largest-sum/description/?envType=study-plan-v2&envId=2024-spring-sprint-100
    (「使……最大值尽可能小」是二分搜索题目常见的问法。)

    给定一个非负整数数组 nums 和一个整数 k ，你需要将这个数组分成 k 个非空的连续子数组。
    设计一个算法使得这 k 个子数组各自和的最大值最小。

    示例 1：

    输入：nums = [7,2,5,10,8], k = 2
    输出：18
    '''
    def splitArray(self, nums: List[int], k: int) -> int:
        def check(nums, target, k):
            cnt, total = 1, 0
            for num in nums:
                if total + num > target:
                    cnt += 1
                    total = num
                else:
                    total += num
            return cnt <= k

        left, right = max(nums), sum(nums)
        while left < right:
            mid = (right - left) // 2 + left
            if check(nums, mid, k):
                right = mid
            else:
                left = mid + 1

        return left

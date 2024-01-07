#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/7 00:20
# @Author  : MinisterYU
# @File    : B_排列求和问题.py
from typing import List


class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        # https://leetcode.cn/problems/arithmetic-slices/
        # 等差数列
        if len(nums) < 3:
            return 0

        dp = [0] * len(nums)

        for i in range(2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i - 1] + 1
        return sum(dp)

        def countSliceEndAt(nums, end):
            # 递归解决
            if end < 2:
                return 0
            count = 0
            if nums[end] - nums[end - 1] == nums[end - 1] - nums[end - 2]:
                count = 1 + countSliceEndAt(nums, end - 1)

            return count

        total_count = 0

        for i in range(len(nums)):
            total_count += countSliceEndAt(nums, i)

        return total_count

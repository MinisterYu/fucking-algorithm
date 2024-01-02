#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/2 14:01
# @Author  : MinisterYU
# @File    : 差分数组.py

from typing import List
from itertools import accumulate


class Solution:

    @staticmethod
    def minSubArrayLen(target: int, nums: List[int]) -> int:
        '''
给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其总和大于等于 target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，
并返回其长度。如果不存在符合条件的子数组，返回 0 。

示例 1：
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
        '''
        res = len(nums) + 1
        preSum = [0] + list(accumulate(nums))
        # for i in range(len(preSum)):
        #     for j in range(i):
        #         if preSum[i] - preSum[j] >= target:
        #             res = min(res, i - j )
        # print(res)

        left = 0  # 左指针
        for right in range(1, len(preSum) + 1):
            while preSum[right] - preSum[left] >= target:
                res = min(res, right - left)
                left += 1

    @staticmethod
    def productExceptSelf(nums):
        n = len(nums)
        left = [1] * n
        right = [1] * n

        for i in range(1, n):
            left[i] = left[i - 1] * nums[i - 1]
        for j in range(n - 2, -1, -1):
            right[i] = right[i + 1] * nums[i + 1]

        ans = [left[i] * right[i] for i in range(n)]
        return ans

    @staticmethod
    def checkSubarraySum(nums: List[int], k: int) -> bool:
        '''
        输入：nums = [23,2,4,6,7], k = 6
        输出：true
        解释：[2,4] 是一个大小为 2 的子数组，并且和为 6 。
        '''
        if len(nums) < 2:
            return False

        preSum = nums[0] % k
        memo = {preSum: 0}
        for i in range(1, len(nums)):
            preSum = (nums[i] + nums[i - 1]) % k
            if not preSum in memo:
                memo[preSum] = i
            else:
                if i - memo[preSum] >= 2:
                    return True
        return False


if __name__ == '__main__':
    # Solution.minSubArrayLen(target=7, nums=[2, 3, 1, 2, 4, 3])
    res = Solution.checkSubarraySum([23, 2, 4, 6, 7], 6)
    print(res)

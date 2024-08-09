#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/27 下午10:33
# @Author  : MinisterYU
# @File    : 递增的三元子序列.py

class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        # https://leetcode.cn/problems/increasing-triplet-subsequence/description/?envType=study-plan-v2&envId=leetcode-75
        '''
        给你一个整数数组 nums ，判断这个数组中是否存在长度为 3 的递增子序列。

        如果存在这样的三元组下标 (i, j, k) 且满足 i < j < k ，使得 nums[i] < nums[j] < nums[k] ，返回 true ；否则，返回 false 。

        示例 1：

        输入：nums = [1,2,3,4,5]
        输出：true
        解释：任何 i < j < k 的三元组都满足题意
        '''
        first = second = max(nums) + 1
        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                return True
        return False
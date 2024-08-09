#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/23 21:07
# @Author  : MinisterYU
# @File    : 最长递增子序列.py
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        '''
        https://leetcode.cn/problems/longest-increasing-subsequence/description/?envType=study-plan-v2&envId=2024-spring-sprint-100
        给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
        子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列
        示例 1：

        输入：nums = [10,9,2,5,3,7,101,18]
        输出：4
        解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
'''

        def bisearch(arr, target):
            l, r = 0, len(arr)
            while l < r:
                mid = l + (r - l) // 2
                if arr[mid] >= target:
                    r = mid
                else:
                    l = mid + 1
            return l
        res = []
        for num in nums:
            pos = bisearch(res, num)
            if pos == len(res):
                res.append(num)
            else:
                res[pos] = num
        # print(res)
        return len(res)

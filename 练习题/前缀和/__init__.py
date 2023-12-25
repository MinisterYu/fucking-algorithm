#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 00:35
# @Author  : MinisterYU
# @File    : __init__.py.py

from typing import List
from itertools import accumulate
from collections import defaultdict

class Solution:

    # URL https://leetcode.cn/problems/binary-subarrays-with-sum/
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        # 给你一个二元数组 nums ，和一个整数 goal ，请你统计并返回有多少个和为 goal 的 非空 子数组。
        n = len(nums)
        presum = [0] + list(accumulate(nums))  # 计算前缀和
        hashmap = defaultdict(int, {0: 1})
        ans = 0

        for i in range(n):
            right = presum[i + 1]
            left = right - goal
            ans += hashmap[left]
            hashmap[right] += 1
        return ans

    # URL https://leetcode.cn/problems/continuous-subarray-sum/
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        #
        if len(nums) < 2:
            return False

        remaind_map = {0: -1}
        remaind = 0
        for i in range(len(nums)):
            remaind = (remaind + nums[i]) % k
            if remaind not in remaind_map:
                remaind_map[remaind] = i
            else:
                if i - remaind_map[remaind] >= 2:
                    return True
        return False


if __name__ == '__main__':
    so = Solution()
    so.numSubarraysWithSum([1, 0, 1, 0, 1], 2)

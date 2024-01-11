#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 16:30
# @Author  : MinisterYU
# @File    : __init__.py.py
from typing import List
import collections


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked
        has_map = collections.defaultdict(list)
        for i in range(len(nums)):
            has_map[nums[i]].append(i)

        keys = sorted(has_map.keys())
        left = 0
        right = len(keys) - 1
        while left <= right:
            if keys[left] + keys[right] > target:
                right -= 1
            elif keys[left] + keys[right] < target:
                left -= 1
            else:
                if keys[left] == keys[right] and len(has_map[keys[left]]) > 1:
                    return has_map[keys[left]][:2]
                elif left != right:
                    return has_map[keys[left]][0], has_map[keys[right]][0]
        return 0, 0

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        print(intervals)
        res = [intervals[0]]
        i = 1

        while i < len(intervals):
            if res[-1][1] < intervals[i][0]:
                res.append(intervals[i])
            else:
                res[-1][0] = min(res[-1][0], intervals[i][0])
                res[-1][1] = max(res[-1][1], intervals[i][1])
            i += 1

        print(res)

    def rotate(self, nums: List[int], k: int):
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n
        if k == 0:
            return nums
        nums = nums[n - k:] + nums[: n - k]
        nums.reverse()
        print(nums)


if __name__ == '__main__':
    so = Solution()
    # twoSum
    # res = so.twoSum([2, 5, 5, 11], 10)
    # print(res)
    # a = sorted('ddccaa')
    # print(a)
    # so.merge([[1, 4], [0, 2], [3, 5]])
    so.rotate([1, 2, 3, 4, 5], 2)
    num = [1,2,3]
    num.reverse()
    print()

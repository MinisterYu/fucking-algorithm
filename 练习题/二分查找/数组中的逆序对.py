#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 20:52
# @Author  : MinisterYU
# @File    : 数组中的逆序对.py
# https://www.nowcoder.com/practice/96bd6684e04a44eb80e6a68efc0ec6c5?tpId=295&tqId=1499549&ru=%2Fpractice%2F71cef9f8b5564579bf7ed93fbe0b2024&qru=%2Fta%2Fformat-top101%2Fquestion-ranking&sourceUrl=%2Fexam%2Foj
from typing import List
class Solution:
    def InversePairs(self, nums: List[int]) -> int:
        return self.get_pairs(nums)[0] % 1000000007

    def get_pairs(self, nums):
        if len(nums) <= 1:
            return 0, nums
        mid = len(nums) // 2

        left_pairs, left_sort = self.get_pairs(nums[:mid])

        right_pairs, right_sort = self.get_pairs(nums[mid:])
        list_sort = []
        pairs_count = 0
        i, j = 0, 0

        while i < len(left_sort) and j < len(right_sort):
            if left_sort[i] <= right_sort[j]:
                list_sort.append(left_sort[i])
                i += 1
            else:
                list_sort.append(right_sort[j])
                j += 1
                pairs_count += len(left_sort) - i

        list_sort.extend(left_sort[i:])
        list_sort.extend(right_sort[j:])

        return left_pairs + right_pairs + pairs_count, list_sort

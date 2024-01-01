#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/1 23:33
# @Author  : MinisterYU
# @File    : 快慢指针.py
from typing import List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # https://leetcode.cn/problems/remove-duplicates-from-sorted-array/
        if not nums:
            return 0
        slow, fast = 0, 1
        while fast <= len(nums) - 1:
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1

    def removeElement(self, nums: List[int], val: int) -> int:
        # https://leetcode.cn/problems/remove-element/
        slow = fast = 0
        while fast <= len(nums) - 1:
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

        return slow
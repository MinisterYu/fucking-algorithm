#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/8 下午10:54
# @Author  : MinisterYU
# @File    : 寻找重复数.py
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        '''https://leetcode.cn/problems/find-the-duplicate-number/description/
        给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
        假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。
        你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                slow = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return slow
        return None
        '''
        slow = nums[0]
        fast = nums[0]

        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        slow = nums[0]

        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow
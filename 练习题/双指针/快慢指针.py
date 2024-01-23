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

    def numSubarrayProductLessThanK(self, nums, k):
        # TODO 713. 乘积小于 K 的子数组 | 给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。
        # https://leetcode.cn/problems/subarray-product-less-than-k/
        if k <= 1:
            return 0

        ans = 0
        left = 0
        count = 1
        for right, value in enumerate(nums):
            count *= value
            while count >= k:
                count /= nums[left]
                left += 1
            ans += right - left + 1
        return ans

    def minSubArrayLen(self, target, nums):
        # TODO 209. 长度最小的子数组 : 固定左指针，右指针一直到边界，再缩小左指针
        # https://leetcode.cn/problems/minimum-size-subarray-sum/description/
        left = 0
        ans = len(nums) + 1
        count = 0
        for right, value in enumerate(nums):
            count += value
            while count >= target:
                ans = min(ans, right - left + 1)
                count -= nums[left]
                left += 1
        return ans if ans < len(nums) + 1 else 0

    def removeDuplicates(self, nums: List[int], k=2) -> int:
        # 删除有序数组中的重复项， 只保留至多 k 个相同数字
        # https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/?envType=study-plan-v2&envId=top-interview-150
        n = len(nums)
        if n <= k:
            return n

        slow = fast = k
        while fast < n:
            if nums[slow - k] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

        print(nums[:slow])
        return slow


if __name__ == '__main__':
    so = Solution()
    so.removeDuplicates([1,2,3], k=1)

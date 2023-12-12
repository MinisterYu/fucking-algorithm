#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 17:27
# @Author  : MinisterYU
# @File    : __init__.py.py


class Solution(object):

    # TODO 167. 两数之和 II - 输入有序数组，双指针，需要排序
    def twoSum(self, numbers, target):
        left = 0
        right = len(numbers) - 1

        while left <= right:
            if numbers[left] + numbers[right] < target:
                left += 1
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                return [left + 1, right + 1]

    # TODO 15. 三数之和 - 输入有序数组，双指针 （去重）
    def threeSum(self, nums):
        ans = []
        nums.sort()
        n = len(nums)

        for i in range(n - 2):
            x = nums[i]
            # 去重操作
            if i > 0 and x == nums[i - 1]:
                continue

            y = i + 1
            z = n - 1
            while y < z:
                res = x + nums[y] + nums[z]
                if res > 0:
                    z -= 1
                elif res < 0:
                    y += 1
                else:
                    ans.append([x, nums[y], nums[z]])
                    y += 1
                    while y < z and nums[y] == nums[y - 1]:
                        y += 1
                    z -= 1
                    while y < z and nums[z] == nums[z + 1]:
                        z -= 1
        return ans

    # TODO 11. 盛最多水的容器： 双向指针
    def maxArea(self, height):
        ans = 0
        left = 0
        right = len(height) - 1
        while left < right:
            res = (right - left) * min(height[left], height[right])
            ans = max(res, ans)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans

    # TODO 42. 接雨水， 算出前序和后序遍历的最大值，取最小，然后减去高度就等于雨水
    def trap(self, height):
        n = len(height)
        pre = [0] * n
        pre[0] = height[0]
        for i in range(1, n):
            pre[i] = max(pre[i - 1], height[i])


        post = [0] * n
        post[-1] = height[-1]
        for i in range(n - 2, -1, -1):
            post[i] = max(post[i + 1], height[i])

        ans = 0
        for x, y, z in zip(height, pre, post):
            ans += min(y, z) - x
        return ans

    # TODO 209. 长度最小的子数组 : 固定左指针，右指针一直到边界，再缩小左指针
    def minSubArrayLen(self, target, nums):
        left = 0
        ans = len(nums) + 1
        count = 0
        for right, value in enumerate(nums):
            count += value
            while count >= target:
                ans = min(ans, right -left + 1)
                count -= nums[left]
                left += 1
        return ans if ans < len(nums) + 1 else 0


    #TODO 713. 乘积小于 K 的子数组 | 给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。
    def numSubarrayProductLessThanK(self, nums, k):
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

    #TODO 3. 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s):
        n = len(s)
        if n <= 1:
            return n

        max_len = 0  # 最长子串的长度
        left = 0  # 滑动窗口的左边界
        seen = {}  # 记录字符最后一次出现的位置

        for right, value in enumerate(s):
            if value in seen and seen[value] >= left:
                left = seen[value] + 1

            seen[value] = right
            max_len = max(max_len, right - left + 1)

        return max_len

def lengthOfLongestSubstring( s):
        n = len(s)
        if n <= 1:
            return n

        max_len = 0  # 最长子串的长度
        left = 0  # 滑动窗口的左边界
        seen = {}  # 记录字符最后一次出现的位置

        for right, value in enumerate(s):
            if value in seen and seen[value] >= left:
                left = seen[value] + 1 # 找到 left的位置 + 1

            seen[value] = right
            max_len = max(max_len, right - left + 1)

        return max_len

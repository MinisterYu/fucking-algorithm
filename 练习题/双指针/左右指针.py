#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/1 23:34
# @Author  : MinisterYU
# @File    : 左右指针.py
from typing import List


class Solution:
    def maxArea(self, height):
        # TODO 11. 盛最多水的容器： 双向指针
        # https://leetcode.cn/problems/container-with-most-water/description/
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

    def reverseString(self, s: List[str]) -> None:
        # todo 翻转字符串
        # https://leetcode.cn/problems/reverse-string/
        left = 0
        right = len(s) - 1
        while left <= right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

    def longestPalindrome(self, s: str) -> str:
        # TODO 最长回文子串
        # https://leetcode.cn/problems/longest-palindromic-substring/
        def isPalindrome(left, right, s):
            while left >= 0 and right <= len(s) - 1 and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1: right]

        res = ""
        for i in range(len(s)):
            res1 = isPalindrome(i, i, s)
            res2 = isPalindrome(i, i + 1, s)

            res = res1 if len(res1) > len(res) else res
            res = res2 if len(res2) > len(res) else res
        return res

    def twoSum(self, numbers, target):
        # TODO 167. 两数之和 II - 输入有序数组，双指针，需要排序
        # https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/description/
        left = 0
        right = len(numbers) - 1

        while left <= right:
            if numbers[left] + numbers[right] < target:
                left += 1
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                return [left + 1, right + 1]

    def threeSum(self, nums):
        # TODO 15. 三数之和 - 输入有序数组，双指针 （去重）
        #
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

    # TODO 3. 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s):
        from collections import defaultdict
        if len(s) <= 1:
            return len(s)

        visited = defaultdict(int)
        left = 0
        ans = 0
        for right in range(len(s)):
            visited[s[right]] += 1

            while visited[s[right]] > 1:
                visited[s[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans

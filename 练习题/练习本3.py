#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/9 上午11:24
# @Author  : MinisterYU
# @File    : 练习本3.py
from typing import List, Optional


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        '''
        给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
        https://leetcode.cn/problems/median-of-two-sorted-arrays/description/
        '''
        m, n = len(nums1), len(nums2)
        temp = []

        def dfs(i, j):
            if i == m:
                temp.extend(nums2[j:])
                return
            if j == n:
                temp.extend(nums1[i:])
                return

            if nums1[i] < nums2[j]:
                temp.append(nums1[i])
                dfs(i + 1, j)
            else:
                temp.append(nums2[j])
                dfs(i, j + 1)

        dfs(0, 0)
        if len(temp) % 2 == 1:
            return temp[len(temp) // 2]
        return (temp[len(temp) // 2] + temp[(len(temp)) // 2 - 1]) / 2

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # https://leetcode.cn/problems/3sum-closest/
        nums.sort()
        n = len(nums)
        min_diff = float('inf')
        res = 0
        for i in range(n - 2):
            j = i + 1
            k = n - 1
            while j < k:
                total = nums[i] + nums[j] + nums[k]
                if total == target:
                    return total

                diff = abs(total - target)
                if min_diff > diff:
                    min_diff = diff
                    res = total

                if total < target:
                    j += 1
                else:
                    k -= 1
        return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, n - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                k = j + 1
                l = n - 1
                while k < l:
                    total = nums[i] + nums[j] + nums[k] + nums[l]
                    if total == target:
                        res.append([nums[i], nums[j], nums[k], nums[l]])

                        k += 1
                        while k < l and nums[k] == nums[k - 1]:
                            k += 1
                        l -= 1
                        while k < l and nums[l] == nums[l + 1]:
                            l -= 1
                    elif total < target:
                        k += 1
                    else:
                        l -= 1
        return res

    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)

        while left < right:
            mid = (right - left) // 2 + left
            if nums[mid] == target:
                return mid

            # if nums[mid] < nums[-1]:
            #     if nums[mid] < target <= nums[-1]:
            #         left = mid + 1
            #     else:
            #         right = mid
            # else:
            #     if nums[0] <= target < nums[mid]:
            #         right = mid
            #     else:
            #         left = mid + 1

            if nums[mid] > nums[left]:
                if nums[left] <= target < nums[mid]:
                    right = mid
                else:
                    left = mid + 1

            else:
                if nums[mid] < target <= nums[right - 1]:
                    left = mid + 1
                else:
                    right = mid
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/
        def bisearch(nums, target):
            left = 0
            right = len(nums)
            while left < right:
                mid = (right - left) // 2 + left
                if nums[mid] == target:
                    return mid
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left

        begin = bisearch(nums, target)
        if begin == len(nums) or nums[begin] != target:
            return [-1, -1]
        end = bisearch(nums, target + 1) - 1
        return [begin, end]

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # https://leetcode.cn/problems/valid-sudoku/
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num != '.':
                    if num in rows[i]:
                        return False
                    rows[i].add(num)
                    if num in cols[j]:
                        return False
                    cols[j].add(num)
                    box_index = (i // 3) * 3 + j // 3
                    if num in boxes[box_index]:
                        return False
                    boxes[box_index].add(num)
        return True


if __name__ == '__main__':
    so = Solution()
    print(so.threeSumClosest([-1, 2, 1, -4], 1))
    s = []
    s.reverse()

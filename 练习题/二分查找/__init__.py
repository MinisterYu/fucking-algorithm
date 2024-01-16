#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/5 17:37
# @Author  : MinisterYU
# @File    : __init__.py.py

nums = [3, 6, 6, 8, 8, 12]
target = 6


# todo 左闭右闭
def lower_bound(nums, target):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (right - left) // 2 + left
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


print(lower_bound(nums, target=6))
print(lower_bound(nums, target=7) - 1)


# todo 左闭右开
def lower_bound2(nums, target):
    left = 0
    right = len(nums)
    while left < right:
        mid = (right - left) // 2 + left
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


print(lower_bound2([1, 2, 3, 4, 5, 6], 3))


# todo >= target lower_bound(nums, target)
# todo >  target lower_bound(nums, target +1 )
# todo <  target lower_bound(nums, target) -1
# todo =< target lower_bound(nums, target+1) -1


# todo 最长公共前缀
def longest_common_prefix(strs):
    if not strs:
        return ""

    def find_prefix(prefix, strs):
        return all(x.startswith(prefix) for x in strs)

    n = min(len(x) for x in strs)

    left = 0
    right = n

    while left < right:
        mid = left + (right - left) // 2
        if find_prefix(strs[0][:mid + 1], strs[1:]):
            left = mid + 1
        else:
            right = mid

    return strs[0][:left]


# TODO 34. 在排序数组中查找元素的第一个和最后一个位置
class Solution(object):
    def searchRange(self, nums, target):
        start = self.middle(nums, target)
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        end = self.middle(nums, target + 1) - 1
        return [start, end]

    def middle(self, nums, target):
        left = 0
        right = len(nums)
        while left < right:
            mid = (right - left) // 2 + left
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid

        return left

    # todo 162. 寻找峰值 - 搜索旋转排序数组
    def findPeakElement(self, nums):
        left = 0
        right = len(nums) - 1

        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1

        return left

    # todo 153. 寻找旋转排序数组中的最小值
    def findMin(self, nums):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (right + left) // 2
            if nums[mid] < nums[mid + 1]:
                right = mid - 1
            else:
                left = mid + 1
        return nums[left]

    # 寻找第K小的元素
    def findKthSmallest(self, nums, k):
        left, right = nums[0], nums[-1]

        while left < right:
            mid = left + (right - left) // 2
            count = 0

            for num in nums:
                if num <= mid:
                    count += 1

            if count < k:
                left = mid + 1
            else:
                right = mid

        return left

    # todo  33. 搜索旋转排序数组
    def search(self, nums, target):
        left = 0
        right = len(nums)
        while left < right:
            mid = (right - left) // 2 + left
            if nums[mid] == target:
                return mid
            # 升序排列里面
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

s1 = set()
s1.intersection()
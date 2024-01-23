#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 21:41
# @Author  : MinisterYU
# @File    : 旋转数组最小数字.py
# https://www.nowcoder.com/practice/9f3231a991af4f55b95579b44b7a01ba?tpId=295&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Foj
class Solution:
    def minNumberInRotateArray(self, nums: List[int]) -> int:
        # write code here
        '''
如果中间位置的元素大于右指针位置的元素，说明最小值在中间位置的右侧，将 left 移动到 mid + 1。
如果中间位置的元素小于右指针位置的元素，说明最小值在中间位置的左侧或者就是中间位置元素本身，将 right 移动到 mid。
如果中间位置的元素等于右指针位置的元素，无法判断最小值在哪一侧，但可以将 right 向左移动一位，缩小查找范围。
        '''
        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) // 2

            if nums[mid] > nums[right]:
                left = mid + 1
            elif nums[mid] < nums[right]:
                right = mid
            else:
                right -= 1

        return nums[left]

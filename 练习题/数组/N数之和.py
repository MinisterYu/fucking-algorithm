#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 23:11
# @Author  : MinisterYU
# @File    : N数之和.py
def nSumTarget(nums, n, start, target):
    sz = len(nums)
    res = []
    if n < 2 or sz < n:
        return res
    if n == 2:
        lo = start
        hi = sz - 1
        while lo < hi:
            left = nums[lo]
            right = nums[hi]
            sum = left + right
            if sum < target:
                while lo < hi and nums[lo] == left:
                    lo += 1
            elif sum > target:
                while lo < hi and nums[hi] == right:
                    hi -= 1
            else:
                res.append([left, right])
                while lo < hi and nums[lo] == left:
                    lo += 1
                while lo < hi and nums[hi] == right:
                    hi -= 1
    else:
        for i in range(start, sz):
            sub = nSumTarget(nums, n - 1, i + 1, target - nums[i])
            for arr in sub:
                arr.append(nums[i])
                res.append(arr)
            while i < sz - 1 and nums[i] == nums[i + 1]:
                i += 1
    return res

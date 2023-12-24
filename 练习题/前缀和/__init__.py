#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 00:35
# @Author  : MinisterYU
# @File    : __init__.py.py

def checkSubarraySum(nums, k):
    prefix_sum = [0]
    for num in nums:
        prefix_sum.append(prefix_sum[-1] + num)

    remainder_map = {0: -1}
    for i in range(len(prefix_sum)):
        curr_sum = prefix_sum[i]
        curr_remainder = curr_sum % k
        if curr_remainder in remainder_map:
            if i - remainder_map[curr_remainder] >= 2:
                return True
        else:
            remainder_map[curr_remainder] = i

    return False


# print(checkSubarraySum([23, 2, 4, 6, 7], 6))

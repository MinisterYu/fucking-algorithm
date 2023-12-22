#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 16:55
# @Author  : MinisterYU
# @File    : 删除数组中重复项.py
def removeDuplicates(nums):
    k = 2
    slow = 0
    for num in nums:
        # 与当前写入的位置前面的第 k 个元素进行比较，不相同则保留
        if slow < k or nums[slow - k] != num:
            nums[slow] = num
            slow += 1
    return slow

nums = [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3]
print(removeDuplicates(nums))

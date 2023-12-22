#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 22:09
# @Author  : MinisterYU
# @File    : 打家劫舍.py
from typing import List

'''
链接：https://leetcode.cn/problems/Gu0c2T/
'''


class Solution:
    # TODO ------------- 打家劫舍 -----------------
    # TODO 198. 打家劫舍: 状态转移，注意初始化，和转移方程
    '''
           定义：dp[i] : i 是 nums 第i个节点， dp[i]是 求的最大值
           递推公式： dp[i] = max(dp[i-2] + nums[i], dp[i - 1])
           初始化：
           dp = [0] * len(nums)
           dp[0] = nums[0]
           dp[1] = max(nums[0], nums[1])
           遍历顺序： 从左往右即可
    '''

    def rob(self, nums):
        dp = [0] * (len(nums) + 1)
        dp[1] = nums[0]
        for i in range(2, len(nums) + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[-1]

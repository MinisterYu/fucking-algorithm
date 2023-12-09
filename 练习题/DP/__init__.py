#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 22:09
# @Author  : MinisterYU
# @File    : __init__.py.py


'''
啥都没有
'''


def findTargetSumWays(nums, target):
    n = len(nums)
    # 初始化的值随提变动
    dp = [[0] * (target + 1)]
    #
    dp[0] = 0

    # 求满足target的组合数有多少
    for i in range(len(nums)):
        for j in range(target, nums[i] - 1, -1):
            dp[j] += dp[j - nums[i]]

    # 求nums里面组合尽量满足 target 的最大值
    for i in range(len(nums)):
        for j in range(target, nums[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])

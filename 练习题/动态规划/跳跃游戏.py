#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 22:18
# @Author  : MinisterYU
# @File    : 跳跃游戏.py
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # https://leetcode.cn/problems/jump-game/
        # 能不能跳到最后一个位置
        dp = [False] * len(nums)
        jump_size = nums[0]
        dp[0] = True
        for i in range(1, len(nums)):
            # 如果能跳到位置 i
            if jump_size >= i:
                # 更新可以跳到的位最远位置
                jump_size = max(jump_size, nums[i] + i)
                dp[i] = True
        return dp[-1]

    def jump(self, nums: List[int]) -> int:
        # https://leetcode.cn/problems/jump-game-ii/
        # 跳到最后一个位子的最少跳跃次数
        n = len(nums)
        dp = [n + 1] * n
        dp[0] = 0
        j = 0
        for i in range(1, n):
            # 当 j + nums[j] < i 时，说明位置 j 无法到达位置 i，我们将 j 增加 1，直到找到一个能够到达位置 i 的最远位置。
            while j + nums[j] < i:
                j += 1
            # 我们将位置 i 的最小跳跃次数设为 dp[j] + 1，即从位置 j 跳到位置 i 需要的最小跳跃次数。
            dp[i] = dp[j] + 1
        return dp[-1]


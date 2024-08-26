#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/22 下午10:10
# @Author  : MinisterYU
# @File    : 交替字符串.py
class Solution:
    # https://leetcode.cn/problems/interleaving-string/
    # 给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        定义 dp，其中 dp[i][j] 表示 s1 的前 i 个字符和 s2 的前 j 个字符是否可以交错组成 s3 的前 i+j 个字符。
        如果 s1 的第 i 个字符和 s3 的第 i+j 个字符相等，那么 dp[i][j] 的值取决于 dp[i-1][j] 的值；
        如果 s2 的第 j 个字符和 s3 的第 i+j 个字符相等，那么 dp[i][j] 的值取决于 dp[i][j-1] 的值。
        基于上述递推关系，我们可以得到以下动态规划的状态转移方程：'''
        m, n = len(s1), len(s2)
        if m + n != len(s3):
            return False

        dp = [[False] * (n + 1) for i in range(m + 1)]
        dp[0][0] = True

        for i in range(m + 1):
            for j in range(n + 1):
                if i > 0 and s1[i - 1] == s3[i + j - 1] and dp[i - 1][j]:
                    dp[i][j] = True
                if j > 0 and s2[j - 1] == s3[i + j - 1] and dp[i][j - 1]:
                    dp[i][j] = True
        return dp[-1][-1]
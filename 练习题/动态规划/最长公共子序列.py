#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 10:20
# @Author  : MinisterYU
# @File    : 最长公共子序列.py
class Solution:
    def LCS(self, s1, s2):
        # write code here
        m, n = len(s1), len(s2)
        # 创建一个二维数组 dp，dp[i][j] 表示 str1 的前 i 个字符和 str2 的前 j 个字符的最长公共子序列的长度
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    # 如果当前字符相等，则最长公共子序列的长度加一
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    # 如果当前字符不相等，则最长公共子序列的长度为前一个状态中的最大值
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # dp[m][n] 为找到最长公共子序列的路径
        # 从 dp[m][n] 开始回溯，构造最长公共子序列
        i, j = m, n
        lcs = []
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                # 如果当前字符相等，则将该字符添加到最长公共子序列中
                lcs.append(s1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                # 如果上方的值大于左方的值，则向上移动
                i -= 1
            else:
                # 否则向左移动
                j -= 1

        # 将最长公共子序列反转并转换为字符串
        lcs.reverse()
        lcs_str = ''.join(lcs)

        if len(lcs_str) == 0:
            return "-1"
        else:
            return lcs_str


def LCS(str1, str2):
    # write code here
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    max_length = 0
    end_position = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_position = j
    print(max_length)
    longest_substring = str2[end_position - max_length: end_position]
    print(longest_substring)

LCS("1AB2345CD", "12345EF")

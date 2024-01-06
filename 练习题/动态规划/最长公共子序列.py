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

    # TODO ------------- 子序列 -----------------

    # TODO 300. 最长子序列 1维
    def lengthOfLIS(self, nums):
        dp = [1] * (len(nums))

        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:  # 判断条件
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

    # TODO 714. 最长连续子序列  1维
    def findLengthOfLCIS(self, nums):

        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:  # 判断条件
                dp[i] = dp[i - 1] + 1

        return max(dp)

    # TODO 718. 最长连续子序列  2维
    def findLength(self, nums1, nums2):

        dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
        ans = 0
        max_i = []
        for i in range(1, len(nums1) + 1):
            for j in range(1, len(nums2) + 1):
                if nums1[i - 1] == nums2[j - 1]:  # 判断条件
                    dp[i][j] = dp[i - 1][j - 1] + 1

            ans = max(ans, max(dp[i]))

        ''' dp:
        [0, 0, 0, 0, 0, 0]
        [0, 1, 1, 1, 1, 1]
        [0, 1, 2, 2, 2, 2]
        [0, 1, 2, 3, 3, 3]
        [0, 1, 2, 3, 4, 4]
        [0, 1, 2, 3, 4, 5]
        '''
        trace = []
        for i in range(len(nums1) + 1, 0, -1):
            for j in range(len(nums2) + 1, 0, -1):
                if dp[i][j] == ans:
                    trace.append([i, j])



        return ans

    # TODO 1143. 最长子序列 2维 | "ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
    def longestCommonSubsequence(self, text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:  # 判断条件
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    # TODO 53. 最大子序和 | 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    def maxSubArray(self, nums) -> int:

        dp = [0] * (len(nums))

        dp[0] = nums[0]

        for i in range(1, len(nums)):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

    # TODO 回文串：(最长连续
    def longestPalindrome_连续(self, s):
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = ""

        for i in range(n - 1, -1, -1):  # 从下往上遍历
            for j in range(i, n):  # 从左往右遍历
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    # dp[i][j] 定义 s字符串中，从i到j的最长回文子串
                    # dp[i][j] = s[i] 等于 s[j] 且 （ 左右间隔小于2，（aa,aba） 或者上一个状态为一个回文串 ）
                    dp[i][j] = True
                if dp[i][j] and j - i + 1 > len(res):
                    res = s[i:j + 1]

        return res

    def longestPalindrome_子序列(self, s):
        n = len(s)
        dp = [[0] * n for _ in range(n)]

        for i in range(len(s)):  # 初始化
            dp[i][i] = 1

        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])

        return dp[0][n - 1]  # 取右上角的那个，因为是从左往右，从下往上在遍历

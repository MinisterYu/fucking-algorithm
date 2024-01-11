#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 10:05
# @Author  : MinisterYU
# @File    : 动态二刷.py
from typing import List


class Solution:
    def countVowelStrings(self, n: int) -> int:
        # https://leetcode.cn/problems/count-sorted-vowel-strings/
        '''
        给你一个整数 n，请返回长度为 n 、仅由元音 (a, e, i, o, u) 组成且按 字典序排列 的字符串数量。
        字符串 s 按 字典序排列 需要满足：对于所有有效的 i，s[i] 在字母表中的位置总是与 s[i+1] 相同或在 s[i+1] 之前。
        '''
        dp = [[0] * 6 for _ in range(n)]
        for i in range(1, 6):
            dp[0][i] = 1

        for i in range(1, n):
            for j in range(1, 6):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        print(sum(dp[-1]))
        return sum(dp[-1])

    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = []
        path = []

        def backtrack(index):
            if index == n:
                return

            for i in range(index, n):
                if s[index: i + 1] == s[index: i + 1][::-1]:
                    count.append(f'{index}:{i + 1}')
                    backtrack(i + 1)

        backtrack(0)
        print(count)

    def getWordsInLongestSubsequence(self, n: int, words: List[str], groups: List[int]) -> List[str]:
        # https://leetcode.cn/problems/longest-unequal-adjacent-groups-subsequence-i/
        # 给你一个整数 n 和一个下标从 0 开始的字符串数组 words ，和一个下标从 0 开始的 二进制 数组 groups ，两个数组长度都是 n 。
        # 请你返回一个字符串数组，它是下标子序列 依次 对应 words 数组中的字符串连接形成的字符串数组。如果有多个答案，返回任意一个。
        dp = [[0] * 6 for _ in range(n)]
        for i in range(1, 6):
            dp[0][i] = 1

        for i in range(1, n):
            for j in range(1, 6):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return sum(dp[-1])

    def countArrangement(self, n: int) -> int:
        # https://leetcode.cn/problems/beautiful-arrangement/
        '''
        假设有从 1 到 n 的 n 个整数。用这些整数构造一个数组 perm（下标从 1 开始），只要满足下述条件 之一 ，该数组就是一个 优美的排列 ：
        perm[i] 能够被 i 整除
        i 能够被 perm[i] 整除
        给你一个整数 n ，返回可以构造的 优美排列 的 数量 。
        '''
        self.count = 0
        self.visited = [False] * n

        def backtrack(index):
            if index == n + 1:
                self.count += 1
                return
            for i in range(1, n + 1):
                if not self.visited[i - 1] and (i % index == 0 or index % i == 0):
                    self.visited[i - 1] = True
                    backtrack(index + 1)
                    self.visited[i - 1] = False

        backtrack(1)
        return self.count

    def change(self, amount: int, coins: List[int]) -> int:
        # https://leetcode.cn/problems/coin-change-ii/
        # 完全背包，组合
        dp = [0] * (amount + 1)

        dp[0] = 1
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] = dp[j] + dp[j - coins[i]]
        return dp[-1]

    def longestPalindromeSubseq(self, s: str) -> int:
        # https://leetcode.cn/problems/longest-palindromic-subsequence/
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        # n = len(s)
        # dp = [[0] * n for _ in range(n)]
        # for i in range(n - 1, -1, -1):
        #     for j in range(i, n):
        #         if i == j:
        #             dp[i][j] = 1
        #         elif s[i] == s[j]:
        #             dp[i][j] = dp[i + 1][j - 1] + 2
        #         else:
        #             dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

        for i in dp:
            print(i)


if __name__ == '__main__':
    so = Solution()
    # so.countVowelStrings(2)
    # so.countSubstrings('aaa')
    so.longestPalindromeSubseq('bbbab')

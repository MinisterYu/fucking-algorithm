#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/4 09:52
# @Author  : MinisterYU
# @File    : 最值问题.py
import functools


class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # https://leetcode.cn/problems/longest-valid-parentheses/
        '''
        示例 1：
        输入：s = "(()"
        输出：2
        解释：最长有效括号子串是 "()"

        示例 2：
        输入：s = ")()())"
        输出：4
        解释：最长有效括号子串是 "()()"
        '''
        # 栈解决
        stack = []
        max_len = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)  # 记录索引
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    max_len = max(max_len, i - stack[-1])
        return max_len

        # 动归解决
        dp = [0] * len(s)  # dp[i] 为以 [i]结尾的最长有效括号数
        max_length = 0
        for i in range(1, len(s)):
            if s[i] == '(':  # ()(
                dp[i] = 0
            elif s[i] == ')':
                if s[i - 1] == '(':  # ()( -> )
                    dp[i] = s[i - 2] + 2 if i >= 2 else 2

                elif (s[i - 1] == ')'  # (..()-> )
                      and i - dp[i - 1] > 0
                      and s[i - dp[i - 1] - 1] == '('  # 检查前一个右括号对应的最长有效括号子串的前一个字符是不是(
                ):
                    # 前一个最长有效括号字符串 + 2
                    dp[i] = dp[i - 1] + 2 \
                            + (dp[i - dp[i - 1] - 2] if i - dp[i - 1] >= 2 else 0)  # 再加上匹配的左括号的签名的最长子串

                max_length = max(max_length, dp[i])

        return max_length

    def minDistance(self, word1, word2):
        # https://leetcode.cn/problems/edit-distance/
        # 递归解决

        @functools.lru_cache(None)
        def traverse(i, j):
            if len(word1) == i:
                return len(word2) - j
            if len(word2) == j:
                return len(word1) - i

            if word1[i] == word2[j]:
                return traverse(i + 1, j + 1)
            insert = traverse(i, j + 1)
            delete = traverse(i + 1, j)
            replace = traverse(i + 1, j + 1)

            return min(insert, delete, replace) + 1
        return traverse(0, 0)

        # 动归解决
        m, n = len(word1), len(word2)
        # dp[i][j] 表示将 word1 的前 i 个字符转换为 word2 的前 j 个字符所需的最少操作数。
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # for i in dp:
        #     print(i)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        # print('----')
        # for i in dp:
        #     print(i)

        return dp[-1][-1]


if __name__ == '__main__':
    so = Solution()
    so.minDistance("horse", "ros")

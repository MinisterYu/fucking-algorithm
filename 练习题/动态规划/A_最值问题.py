#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/4 09:52
# @Author  : MinisterYU
# @File    : A_最值问题.py
import functools
from typing import List


class Solution:

    def minFlipsMonoIncr(self, s: str) -> int:
        # https://leetcode.cn/problems/cyJERH/?envType=study-plan-v2&envId=coding-interviews-special
        # 翻转最少次数使二进制字符串变成递增
        '''
        其中 dp0[i] 表示将 s[i] 之前的字符全部翻转为 '0' 所需的最小翻转次数，dp1[i] 表示将 s[i] 之前的字符全部翻转为 '1' 所需的最小翻转次数。
        '''
        n = len(s)
        dp_zero = [0] * (n + 1)
        dp_one = [0] * (n + 1)

        for i in range(1, n + 1):
            if s[i - 1] == "0":
                dp_zero[i] = dp_zero[i - 1]
                dp_one[i] = dp_one[i - 1] + 1
            else:
                dp_zero[i] = dp_zero[i - 1] + 1
                dp_one[i] = min(dp_zero[i - 1], dp_one[i - 1])

        return min(dp_zero[-1], dp_one[-1])


    def countSubstrings(self, s: str) -> int:
        # https://leetcode.cn/problems/palindromic-substrings/
        n = len(s)
        count = 0
        dp = [[False] * n for _ in range(n)]  # dp[i][j] 为从i到j为回文子串

        for i in range(n - 1, -1, -1):
            for j in range(i, len(s)):
                if i == j:  # 如果是同个字符，为真
                    dp[i][j] = True
                    count += 1
                if s[i] == s[j] and j - i == 1:  # 如果两个字符相等，且相距只有一个字符，为真
                    dp[i][j] = True
                    count += 1
                if s[i] == s[j] and j - i > 1 and dp[i + 1][j - 1] == True:
                    # 如果两个字符相等，且相距>1，且前一个状态为真，为真
                    dp[i][j] = True
                    count += 1
        return count

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
        if not s: return 0
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
                    dp[i] = dp[i - 2] + 2 if i >= 2 else 2

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

    def maxProduct(self, nums: List[int]) -> int:
        # https://leetcode.cn/problems/maximum-product-subarray/
        '''
        给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
        1 <= nums.length <= 2 * 104
        -10 <= nums[i] <= 10
        nums 的任何前缀或后缀的乘积都 保证 是一个 32-位 整数
        '''
        if not nums:
            return 0

        max_product = nums[0]
        min_product = nums[0]
        result = max_product

        for i in range(1, len(nums)):
            if nums[i] < 0:  # 如果 nums[i] 是负数，交换 max_product 和 min_product 的值
                max_product, min_product = min_product, max_product

            max_product = max(nums[i], max_product * nums[i])
            min_product = min(nums[i], min_product * nums[i])
            result = max(result, max_product)

        return result

    def integerBreak(self, n: int) -> int:
        # https://leetcode.cn/problems/integer-break/
        '''给定一个正整数 n ，将其拆分为 k 个 正整数 的和（ k >= 2 ），并使这些整数的乘积最大化。
        返回 你可以获得的最大乘积 。 n <= 2 '''

        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 0
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)
                # dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        print(dp)
        return max(dp)

    def minCut(self, s: str) -> int:
        # https://leetcode.cn/problems/palindrome-partitioning-ii/
        # 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。返回符合要求的 最少分割次数 。
        n = len(s)
        dp = [[True] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]

        for i in dp:
            print(i)

        dp_f = [float('inf')] * n
        for i in range(n):
            if dp[0][i]:
                dp_f[i] = 0
            else:
                for j in range(i):
                    if dp[j + 1][i]:
                        dp_f[i] = min(dp_f[i], dp_f[j] + 1)

        return dp_f[n - 1]

    def palindrome(self, s: str) -> int:
        # https://leetcode.cn/problems/palindrome-partitioning-ii/
        # 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。返回符合要求的 最少分割次数 。
        n = len(s)
        count = 1
        dp = [['_'] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):  # 从下往上遍历
            for j in range(i, n):  # 从左往右遍历
                if (s[i] == s[j]) and (j - i < 2 or dp[i + 1][j - 1] != '_'):
                    dp[i][j] = s[i]
                    count += 1
            print(dp[i])
        print('------')
        for i in dp:
            print(i)

    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        n = len(books)
        dp = [0] + [float('inf')] * n

        for i in range(1, n + 1):
            total_width = 0
            max_height = 0
            for j in range(i, 0, -1):
                total_width += books[j - 1][0]
                if total_width > shelfWidth:
                    break
                max_height = max(max_height, books[j - 1][1])
                dp[i] = min(dp[i], dp[j - 1] + max_height)

        print(dp[-1])

    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        # https://leetcode.cn/problems/extra-characters-in-a-string/
        word_set = set(dictionary)
        n = len(s)
        dp = [0] + [float("inf")] * n  # dp[i] 表示字符串 s 的前 i 个字符最少剩余的字符数。
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1  # 假设当前字符不在任何单词中
            for j in range(i):
                if s[j:i] in word_set:
                    dp[i] = min(dp[i], dp[j])  # 如果找到一个单词，更新最小剩余字符数
        return dp[n]

    def numDecodings(self, s):
        # 有一种将字母编码成数字的方式：'a'->1, 'b->2', ... , 'z->26'。
        # 现在给一串数字，返回有多少种可能的译码结果
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1 if s[0] != '0' else 0

        for i in range(2, n + 1):
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]
            if '10' <= s[i - 2:i] <= '26':
                dp[i] += dp[i - 2]

        return dp[n]


if __name__ == '__main__':
    so = Solution()
    # so.minDistance("horse", "ros")
    # so.maxProduct([2, 3, -2, 4])
    # so.integerBreak(10)
    # so.palindrome('aacadca')
    so.minHeightShelves(books=[[1, 1], [2, 3], [2, 3], [1, 1], [1, 1], [1, 1], [1, 2]], shelfWidth=4)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/26 上午11:00
# @Author  : MinisterYU
# @File    : 字符串练习本.py
from collections import defaultdict, Counter
from typing import List


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # https://leetcode.cn/problems/longest-substring-without-repeating-characters/
        # 无重复最长字符串的长度
        res = 0
        left = 0
        count = defaultdict(int)
        for right in range(len(s)):
            count[s[right]] += 1
            while count[s[right]] > 1:
                count[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res

    def longestPalindrome(self, s: str) -> str:
        # https://leetcode.cn/problems/longest-palindromic-substring/
        # 找出字符串中最长回文子串
        def find(left, right, s):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1

        res = ''
        res_len = 0
        for i in range(len(s)):
            left, right = find(i, i, s)
            if right - left + 1 > res_len:
                res_len = right - left + 1
                res = s[left:right + 1]
            left, right = find(i, i + 1, s)
            if right - left + 1 > res_len:
                res_len = right - left + 1
                res = s[left:right + 1]
        return res

    def longestCommonPrefix(self, strs: List[str]) -> str:
        # https://leetcode.cn/problems/longest-common-prefix/
        # 最长公共前缀
        strs0 = strs[0]

        left = 0
        right = len(strs0)
        while left < right:
            mid = (right - left) // 2 + left
            if all(s.startswith(strs0[:mid + 1]) for s in strs[1:]):
                left = mid + 1
            else:
                right = mid
        return strs0[:left]

    def letterCombinations(self, digits: str) -> List[str]:
        # https://leetcode.cn/problems/letter-combinations-of-a-phone-number/
        # 电话号码组合
        dialmap = {
            "1": "",
            '2': "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
            "0": ""
        }
        ans = []
        path = []

        def back(index):
            if len(path) == len(digits):
                ans.append(''.join(path[:]))
                return
            number = digits[index]
            for char in dialmap[number]:
                path.append(char)
                back(index + 1)
                path.pop()

        back(0)
        return ans

    def isValid(self, s: str) -> bool:
        # 判断字符串是否是有效的括号
        stack = []
        for char in s:
            if char == '{':
                stack.append('}')
            elif char == '(':
                stack.append(')')
            elif char == '[':
                stack.append(']')
            elif stack[-1] != char or not stack:
                return False
            else:
                stack.pop()
        return len(stack) == 0

    def generateParenthesis(self, n: int) -> List[str]:
        # https://leetcode.cn/problems/generate-parentheses/
        # 括号生成
        ans = []

        def back(left, right, path):
            if len(path) == 2 * n:
                ans.append(path)
                return
            if left < n:
                back(left + 1, right, path + '(')
            if right < left:
                back(left, right + 1, path + ')')

        back(0, 0, '')
        return ans

    def longestValidParentheses(self, s: str) -> int:
        # https://leetcode.cn/problems/longest-valid-parentheses/
        # 有效括号的最长长度
        stack = [-1]
        res = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res

    def minDistance(self, word1: str, word2: str) -> int:
        # https://leetcode.cn/problems/edit-distance/description/
        # 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数
        memo = {}
        m, n = len(word1), len(word2)

        def dfs(i, j):
            if m == i:
                return n - j
            if n == j:
                return m - i
            if (i, j) in memo:
                return memo[(i, j)]

            if word1[i] == word2[j]:
                res = dfs(i + 1, j + 1)
            else:
                res = min(dfs(i + 1, j), dfs(i, j + 1), dfs(i + 1, j + 1)) + 1
            memo[(i, j)] = res
            return res

        return dfs(0, 0)

    def minWindow(self, s: str, t: str) -> str:
        # https://leetcode.cn/problems/minimum-window-substring/
        # 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。
        # 如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
        need = len(t)
        counter_t = Counter(t)
        res = ''
        res_len = len(s) + 1
        left = 0
        for right in range(len(s)):
            char = s[right]
            if char in counter_t:
                if counter_t[char] > 0:
                    need -= 1
                counter_t[char] -= 1

            while need == 0:
                if right - left + 1 < res_len:
                    res_len = right - left + 1
                    res = s[left:right + 1]

                char = s[left]
                if char in counter_t:
                    if counter_t[char] >= 0:
                        need += 1
                    counter_t[char] += 1

                left += 1
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        # https://leetcode.cn/problems/word-search/
        # 搜索单词
        m, n = len(board), len(board[0])
        if m * n < len(word):
            return False
        counter_word = Counter(word)
        counter_board = Counter([board[i][j] for i in range(m) for j in range(n)])

        for char in counter_word:
            if char not in counter_board:
                return False
            if counter_word[char] > counter_board[char]:
                return False
        if counter_board[word[0]] < counter_word[word[-1]]:
            word = word[::-1]
        visited = [[False] * n for _ in range(m)]

        def dfs(i, j, k):
            if k == len(word):
                return True
            if not i < m or not j < n or board[i][j] != word[k] or visited[i][j]:
                return False

            visited[i][j] = True
            res = dfs(i + 1, j, k + 1) or dfs(i, j + 1, k) or dfs(i + 1, j + 1, k + 1)
            visited[i][j] = False
            return res

        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False


if __name__ == '__main__':
    so = Solution()
    print(so.lengthOfLongestSubstring('pwwkew'))

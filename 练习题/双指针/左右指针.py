#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/1 23:34
# @Author  : MinisterYU
# @File    : 左右指针.py
from typing import List


class Solution:
    def reverseString(self, s: List[str]) -> None:
        # https://leetcode.cn/problems/reverse-string/
        left = 0
        right = len(s) - 1
        while left <= right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

    def longestPalindrome(self, s: str) -> str:
        # https://leetcode.cn/problems/longest-palindromic-substring/
        def isPalindrome(left, right, s):
            while left >= 0 and right <= len(s) - 1 and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1: right]

        res = ""
        for i in range(len(s)):
            res1 = isPalindrome(i, i, s)
            res2 = isPalindrome(i, i + 1, s)

            res = res1 if len(res1) > len(res) else res
            res = res2 if len(res2) > len(res) else res
        return res

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 17:35
# @Author  : MinisterYU
# @File    : __init__.py

# TODO 最长回文子串
def longestPalindrome(s):
    start, end = 0, 0
    for i in range(len(s)):
        single_left, single_right = find_max(s, i, i)
        double_left, double_right = find_max(s, i, i + 1)
        if single_right - single_left > end - start:
            start, end = single_left, single_right
        if double_right - double_left > end - start:
            start, end = double_left, double_right
    return s[start: end + 1]


def find_max(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return left + 1, right - 1


# 示例用法
s = "abcbbc"
result = longestPalindrome(s)
print(result)


# TODO 最长公共前缀
class Solution:
    def longestCommonPrefix(self, strs) -> str:

        # for i in range(len(strs[0])):
        #     char = strs[0][i]
        #     for s in strs[1:]:
        #         if i >= len(s) or char != s[i]:
        #             return strs[0][:i]
        # return strs[0]

        strs.sort()
        left = 0
        right = len(strs[0])
        while left < right:
            mid = (right - left) // 2 + left
            # 如果都包含，左移
            if all(s.startswith(strs[0][:mid + 1]) for s in strs[1:]):
                left = mid + 1
            else:
                right = mid

        return strs[0][:left]

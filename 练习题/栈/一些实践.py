#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 16:22
# @Author  : MinisterYU
# @File    : 一些实践.py
class Solution:
    def removeStars(self, s: str) -> str:
        # https://leetcode.cn/problems/removing-stars-from-a-string/?envType=study-plan-v2&envId=leetcode-75
        stack = []
        i = 0
        while i < len(s):
            while i < len(s) and stack and s[i] == '*':
                stack.pop()
                i += 1
            if i < len(s) and s[i] != '*':
                stack.append(s[i])
                i += 1

        return ''.join(stack)

if __name__ == '__main__':
    so = Solution()
    s = so.removeStars("erase*****")
    print(s)
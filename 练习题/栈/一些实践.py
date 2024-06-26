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

    def longestValidParentheses(self, s: str) -> int:
        # 计算字符串中的最长有效括号长度
        stack = []
        stack.append(-1)
        max_length = 0

        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if len(stack) == 0:
                    stack.append(i)
                else:
                    max_length = max(max_length, i - stack[-1])

        return max_length

    def checkValidString(self, s: str) -> bool:
        # 678. 有效的括号字符串
        stack = []
        s_stack = []

        for char in s:
            if char == '(':
                stack.append(char)
            elif char == '*':
                stack.append(char)
            else:
                if stack:
                    stack.pop()
                elif s_stack:
                    stack.pop()
                else:
                    return False
        print(stack)
        print(s_stack)
        return True if len(stack) == 0 else False

if __name__ == '__main__':
    s = Solution()
    s.checkValidString('(*)')
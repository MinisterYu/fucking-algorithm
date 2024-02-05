#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 11:23
# @Author  : MinisterYU
# @File    : 括号的问题.py
class Solution:
    def checkValidString(self, s: str) -> bool:
        '''
        给你一个只包含三种字符的字符串，支持的字符类型分别是 '('、')' 和 '*'。请你检验这个字符串是否为有效字符串，如果是有效字符串返回 true 。

        有效字符串符合如下规则：

        任何左括号 '(' 必须有相应的右括号 ')'。
        任何右括号 ')' 必须有相应的左括号 '(' 。
        左括号 '(' 必须在对应的右括号之前 ')'。
        '*' 可以被视为单个右括号 ')' ，或单个左括号 '(' ，或一个空字符串。
        一个空字符串也被视为有效字符串。

        '''
        stack = []
        r_stack = []
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            elif char == '*':
                r_stack.append(i)
            else:
                if stack:
                    stack.pop()
                elif r_stack:
                    r_stack.pop()
                else:
                    return False

        while stack and r_stack:
            if stack[-1] > r_stack[-1]:
                return False

            stack.pop()
            r_stack.pop()

        return True

    def longestValidParentheses(self, s: str) -> int:
        '''
        给出一个长度为 n 的，仅包含字符 '(' 和 ')' 的字符串，计算最长的格式正确的括号子串的长度。

        例1: 对于字符串 "(()" 来说，最长的格式正确的子串是 "()" ，长度为 2 .
        例2：对于字符串 ")()())" , 来说, 最长的格式正确的子串是 "()()" ，长度为 4 .
        '''
        stack = [-1]
        max_length = 0
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    max_length = max(max_length, i - stack[-1])
        return max_length

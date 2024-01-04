#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/4 09:52
# @Author  : MinisterYU
# @File    : 最值问题.py

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
        
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/23 20:23
# @Author  : MinisterYU
# @File    : 移除K个数.py
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        '''https://leetcode.cn/problems/remove-k-digits/?envType=study-plan-v2&envId=2024-spring-sprint-100
        给你一个以字符串表示的非负整数 num 和一个整数 k ，移除这个数中的 k 位数字，使得剩下的数字最小。请你以字符串形式返回这个最小的数字。
        输入：num = "1432219", k = 3
        输出："1219"
        解释：移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219 。
        '''
        stack = []
        for digit in num:
            while k > 0 and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)

        # 如果还有剩余的 k，从末尾移除 k 个数字
        while k > 0:
            stack.pop()
            k -= 1

        # 移除前导零
        while stack and stack[0] == "0":
            stack.pop(0)

        if not stack:
            return "0"
        return "".join(stack)

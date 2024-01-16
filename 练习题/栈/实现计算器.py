#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/29 23:18
# @Author  : MinisterYU
# @File    : 实现计算器.py
import collections


class Solution:
    def calculate(self, s: str) -> int:
        # https://leetcode.cn/problems/basic-calculator/
        # s = "(1+(4+5+2)-3)+(6+8)"
        def helper(s):
            stack = []
            sign = '+'
            num = 0
            while len(s) > 0:
                c = s.popleft()
                if c.isdigit():
                    num = 10 * num + int(c)
                    continue
                # 遇到左括号开始递归计算 num
                if c == '(':
                    num = helper(s)

                if (not c.isdigit() and c != ' ') or len(s) == 0:
                    if sign == '+':
                        stack.append(num)
                    elif sign == '-':
                        stack.append(-num)
                    elif sign == '*':
                        stack[-1] = stack[-1] * num
                    elif sign == '/':
                        # python 除法向 0 取整的写法
                        stack[-1] = int(stack[-1] / float(num))
                    num = 0
                    sign = c
                # 遇到右括号返回递归结果
                if c == ')': break
            return sum(stack)

        print(helper(collections.deque(s)))


if __name__ == '__main__':
    so = Solution()
    so.calculate("1 + 1")

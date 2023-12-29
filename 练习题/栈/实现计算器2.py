#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/29 23:18
# @Author  : MinisterYU
# @File    : 实现计算器.py
import collections


class Solution:
    def calculate(self, s: str) -> int:

        def executor(queue: collections.deque):
            num = 0
            sign = '+'
            stack = []
            while len(queue) > 0:
                char = queue.popleft()
                if char.isdigit():
                    num = num * 10 + int(char)
                if char == '(':  # 不包含括号的时候，看这个
                    executor(queue)
                # 遇到符号了
                if (not char.isdigit() and char != ' ') or len(queue) == 0:
                    if sign == '+':  # 每次计算，取上一次的符号
                        stack.append(num)
                    if sign == '-':  # 每次计算，取上一次的符号
                        stack.append(-num)
                    if sign == '*':  # 每次计算，取上一次的符号
                        stack[-1] = stack[-1] * num
                    if sign == '/':  # 每次计算，取上一次的符号
                        stack[-1] = stack[-1] // num

                    # 记录当前符号
                    sign = char
                    num = 0  # 数字归零
                if char == ')':
                    break
            return sum(stack)

        print(executor(collections.deque(s)))
        # return executor(queue)


if __name__ == '__main__':
    so = Solution()
    so.calculate('1+4/2 * 6')

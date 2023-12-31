#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/29 23:18
# @Author  : MinisterYU
# @File    : 实现计算器.py
import collections


class Solution:
    def calculate(self, s: str) -> int:
        self.memo = []

        def executor(queue: collections.deque):
            num = 0
            sign = '+'
            stack = []
            while queue:
                char = queue.popleft()
                if char.isdigit():
                    num = num * 10 + int(char)
                if char == '(':  # 不包含括号的时候，看这个
                    num = executor(queue)
                # 遇到符号了
                # if (not char.isdigit() and char != ' ') or not queue:
                if char in '+-*/()' or not queue:
                    self.memo.append(sign)
                    # print(f'char: {char} in "+-" is {char in "+-"}')
                    # if char in '+-' or not queue:
                    if sign == '+':  # 每次计算，取上一次的符号
                        stack.append(num)
                    if sign == '-':  # 每次计算，取上一次的符号
                        stack.append(-num)
                    if sign == '*':  # 每次计算，取上一次的符号
                        stack[-1] = stack[-1] * num
                    if sign == '/':  # 每次计算，取上一次的符号
                        # stack[-1] = stack[-1] // num if stack[-1] >= 0 else  stack[-1] // num + 1
                        stack[-1] = int(stack[-1] / float(num))

                    # 记录当前符号
                    sign = char

                    num = 0  # 数字归零
                if char == ')':
                    break

            return sum(stack)

        s = executor(collections.deque(s))
        print(self.memo)
        print(s)


if __name__ == '__main__':
    so = Solution()
    s = so.calculate("(( (1*3) - 2 ) * ( 2+ 2)) /2 ")

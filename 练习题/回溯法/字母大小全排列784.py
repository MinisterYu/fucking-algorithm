#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/18 10:47
# @Author  : MinisterYU
# @File    : 字母大小全排列784.py
from typing import *

'''
给定一个字符串 s ，通过将字符串 s 中的每个字母转变大小写，我们可以获得一个新的字符串。
返回 所有可能得到的字符串集合 。以 任意顺序 返回输出。

示例 1：
输入：s = "a1b2"
输出：["a1b2", "a1B2", "A1b2", "A1B2"]
'''


class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        result = []
        path = []
        self.backtrack(0, s, result, path)
        return result

    def backtrack(self, start: int, string: str, result: list, path: list):
        if start == len(string):
            result.append(''.join(path[:]))
            return

        if string[start].isalpha():
            # path.append()
            self.backtrack(start + 1, string, result, path + [string[start].upper()])

            # path.append(string[start].lower())
            self.backtrack(start + 1, string, result, path + [string[start].lower()])
        else:
            # path.append(string[start])
            self.backtrack(start + 1, string, result, path + [string[start]])


if __name__ == '__main__':
    solution = Solution()
    res = solution.letterCasePermutation("a1b2")
    print(res)

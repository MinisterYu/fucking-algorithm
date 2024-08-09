#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/27 下午10:47
# @Author  : MinisterYU
# @File    : 压缩字符串.py
from typing import List

class Solution:
    def compress(self, chars: List[str]) -> int:
        # https://leetcode.cn/problems/string-compression/?envType=study-plan-v2&envId=leetcode-75
        '''
        输入：chars = ["a","a","b","b","c","c","c"]
        输出：返回 6 ，输入数组的前 6 个字符应该是：["a","2","b","2","c","3"]
        解释："aa" 被 "a2" 替代。"bb" 被 "b2" 替代。"ccc" 被 "c3" 替代。
        '''
        n = len(chars)
        i = 0  # 定位
        j = 0  # 游标
        while j < n:
            chars[i] = chars[j]
            count = 1
            while j + 1 < n and chars[i] == chars[j + 1]:
                count += 1
                j += 1

            if count > 1:
                count = str(count)
                for digit in count:
                    i += 1
                    chars[i] = digit
            i += 1
            j += 1
        print(chars[:i])
        return i

    def isSubsequence(self, s: str, t: str) -> bool:
        j = 0
        for i in range(len(s)):
            if s[i] == t[i]:
                j += 1
            elif j == len(t) - 1:
                return True

        return False

if __name__ == '__main__':
    so = Solution()
    so.isSubsequence()
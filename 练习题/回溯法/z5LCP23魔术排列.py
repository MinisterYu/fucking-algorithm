#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 11:11
# @Author  : MinisterYU
# @File    : z5LCP23魔术排列.py
from typing import List


# https://leetcode.cn/problems/er94lq/description/

class Solution:
    def isMagic1(self, target: List[int]) -> bool:
        n = len(target)

        def help(target, k):
            nums = sorted(target)
            res = []

            while nums:
                nums = nums[1::2] + nums[::2]
                res += nums[:k]
                nums = nums[k:]

            return res == target

        for i in range(1, len(target)):
            if help(target, i):
                return True
        return False

    def isMagic(self, target):
        #
        arr = sorted(target)
        arr = arr[1::2] + arr[0::2]
        print(arr)
        if arr[0] != target[0]:
            return False

        k = 0
        for a, b in zip(arr, target):
            if a == b:
                k += 1
            else:
                break

        while len(target) >= k:
            target = target[k:]
            arr = arr[k:]
            arr = arr[1::2] + arr[0::2]
            if arr[:k] != target[:k]:
                return False

        return True

    def checkInclusion(self, s1: str, s2: str) -> bool:
        from collections import Counter
        n1 = len(s1)
        n2 = len(s2)

        c1 = Counter(s1)
        c2 = Counter(s2[:n1])

        for i in range(n1, n2):
            if c1 == c2:
                return True
            c2[s2[i]] += 1
            c2[s2[i - n1]] -= 1

        return c1 == c2

if __name__ == '__main__':
    so = Solution()
    s = so.checkInclusion("adc", "dcda")
    print(s)
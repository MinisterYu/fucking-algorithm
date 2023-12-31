#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/31 22:02
# @Author  : MinisterYU
# @File    : 寻找右区间.py
from bisect import bisect_left

intervals = [[3, 4], [2, 3], [1, 2]]
res = [-1] * len(intervals)
stack = []
for k, v in enumerate(intervals):
    v.append(k)

intervals.sort(key=lambda x: x[0])
print(intervals)
for interval in intervals:
    while stack and stack[-1][1] <= interval[0]:
        temp = stack.pop()
        res[temp[2]] = interval[2]
    stack.append(interval)

print(res )
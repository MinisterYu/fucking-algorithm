#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/5 17:37
# @Author  : MinisterYU
# @File    : __init__.py.py
def longest_common_prefix(strs):
    if not strs:
        return ""

    def find_prefix(prefix, strs):
        return all(x.startswith(prefix) for x in strs)

    n = min(len(x) for x in strs)

    left = 0
    right = n

    while left < right:
        mid = left + (right - left + 1) // 2
        if find_prefix(strs[0][:mid], strs[1:]):
            left = mid
        else:
            right = mid - 1

    return strs[0][:left]


# nums 是个 m * n 的网格
def find_all(grid):
    m = len(grid)
    n = len((grid[0]))

    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 1:
                continue
            grid[i][j] = grid[i - 1][j] + grid[i][j - 1]

    print(grid[-1][-1])


# find_all([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
# find_all([[0, 1], [0, 0]])


def find_(prices):
    cost = float('inf')
    prof = 0
    for price in prices:
        cost = min(cost, price)
        prof = max(prof, price - cost)
    print(prof)


find_([7, 6, 5, 8]  )

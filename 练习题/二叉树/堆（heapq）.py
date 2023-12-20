#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/19 22:35
# @Author  : MinisterYU
# @File    : 堆（heapq）.py

from heapq import *

nums = [9, 5, 2, 9, 6, 3, 4, 4, 2, 7]
# nums = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
#
heap = []
for i in nums:
    heappush(heap, i)
res = [heappop(heap) for _ in range(len(heap))]
print(res)

print(nlargest(3, nums, key=lambda x: x * -1))
print(nsmallest(3, nums))

portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 21.09},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]

print(nlargest(2, portfolio, key=lambda x: x['shares']))

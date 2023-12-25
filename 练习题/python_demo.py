#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 23:19
# @Author  : MinisterYU
# @File    : python_demo.py
import collections
import itertools
import operator
import functools
import heapq

# accumulate 前缀和
data = [1, 2, 3, 4, 5, 6, 7]
res = [0] + list(itertools.accumulate(data, operator.add))
print(res)

# functools.reduce 迭代计算
res = functools.reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])
print(res)

dict_ = collections.defaultdict(lambda: 100)
dict_[1] = 2
print(dict_)
print(dict_[2])

heap = []
heapq.heappush(heap, [1, 'a'])
heapq.heappush(heap, [1, 'b'])
heapq.heappush(heap, [2, 'a'])
heapq.heappush(heap, [2, 'a'])
print(heap)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/19 12:30
# @Author  : MinisterYU
# @File    : 集合.py
s1 = [1, 2, 3, 4, 5]
s2 = [3, 4, 5, 6, 7]
s3 = [1,2,3]
s1 = set(s1)
s2 = set(s2)
s3 = set(s3)

# 交集
print(s1.intersection(s2))
# 父级
print(s1.issuperset(s3))
# 子集
print(s3.issubset(s1))
# 并集
print(s1.union(s2).union(s3))
# 差集
print(s1.symmetric_difference(s2))

s1 = [1, 2, 3, 4, 5]



a = '1'
print(a.lower())
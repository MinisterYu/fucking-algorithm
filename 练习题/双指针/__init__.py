#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 17:27
# @Author  : MinisterYU
# @File    : __init__.py.py


class Solution(object):




    # TODO 42. 接雨水， 算出前序和后序遍历的最大值，取最小，然后减去高度就等于雨水
    def trap(self, height):
        n = len(height)
        pre = [0] * n
        pre[0] = height[0]
        for i in range(1, n):
            pre[i] = max(pre[i - 1], height[i])

        post = [0] * n
        post[-1] = height[-1]
        for i in range(n - 2, -1, -1):
            post[i] = max(post[i + 1], height[i])

        ans = 0
        for x, y, z in zip(height, pre, post):
            ans += min(y, z) - x
        return ans









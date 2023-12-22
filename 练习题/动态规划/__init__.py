#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 22:09
# @Author  : MinisterYU
# @File    : __init__.py.py

# 原推导式
def testWeightBagProblem(weight, value, capacity):
    goods = len(weight)  # 获取物品的数量

    dp = [[0] * (capacity + 1) for _ in range(goods + 1)]  # 给物品增加冗余维，i = 0 表示没有物品可选

    # 初始化dp数组，默认全为0即可
    # 填充dp数组
    for i in range(1, goods + 1):
        for j in range(1, capacity + 1):
            if j < weight[i - 1]:  # i - 1 对应物品 i
                """
                当前背包的容量都没有当前物品i大的时候，是不放物品i的
                那么前i-1个物品能放下的最大价值就是当前情况的最大价值
                """
                dp[i][j] = dp[i - 1][j]
            else:
                """
                当前背包的容量可以放下物品i
                那么此时分两种情况：
                   1、不放物品i
                   2、放物品i
                比较这两种情况下，哪种背包中物品的最大价值最大
                """
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i - 1]] + value[i - 1])  # i - 1 对应物品 i

    # 打印dp数组
    for arr in dp:
        print(arr)



class Solution(object):

    # TODO 62. 不同路径 ：
    def uniquePaths(self, m, n):
        dp = [[1] * (n) for _ in range(m)]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[-1][-1]

    # TODO 343. 整数拆分:求最大乘积
    def integerBreak(self, n):
        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 0
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[-1]

    # TODO 96. 不同搜索二叉树：求有多少种组合的二叉树
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0] * (n + 1)

        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                print(i, j)
                dp[i] = dp[i] + dp[i - j] * dp[j - 1]

        return dp[-1]



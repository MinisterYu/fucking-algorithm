#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 22:09
# @Author  : MinisterYU
# @File    : 背包问题.py

'''
01背包 ：n种物品，每种只有一个
完全背包 ： n种物品，每种有无限个
'''

# nums = [[1, 15],
#         [2, 25],
#         [2, 30],
#         [5, 35]]
# caps = 6

'''
dp[i][j]: [0:1]之间的物品任取放入容量为j的背包的最大价值

'''

dp = []
w = [2, 3, 4, 5]  # 商品的体积2、3、4、5
v = [3, 4, 5, 6]  # 商品的价值3、4、5、6

caps = 6


# i = len(v)
# j = caps
# dp[i - 1][j] ： 不选当前i的物品 ；  dp[i - 1][j - w[i]] + v[i]：选当前i的物品
# dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i])

# 暴力搜索
def zero_back(caps, w, v):
    n = len(w)  # 获取物品总数

    def dfs(i, c):
        if i < 0:  # 没有物品可选
            return 0
        if w[i] > c:  # 超过剩余容量了
            return dfs(i - 1, c)
        return max(dfs(i - 1, c), dfs(i - 1, c - w[i]) + v[i])

    return dfs(n - 1, caps)


# print(zero_back(caps, w, v))

w = [2, 3, 4, 5]  # 商品的体积2、3、4、5
v = [3, 4, 5, 6]  # 商品的价值3、4、5、6

caps = 6


def beibao(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1])
            else:
                dp[i][j] = dp[i - 1][j]

    # for i in dp:
    #     print(i)
    return dp[-1][-1]


weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 8

max_value = beibao(weights, values, capacity)


# print("Maximum value:", max_value)


def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j:
                # dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1] )
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1])
            else:
                dp[i][j] = dp[i - 1][j]
    # for i in dp:
    #     print(i)
    # return dp[n][capacity]


max_value = knapsack(weights, values, capacity)


# print("Maximum value:", max_value)


def beibao2(weights, values, capacity):
    goods = len(weights)

    dp = [[0] * (capacity + 1) for _ in range(goods + 1)]

    for i in range(1, goods + 1):
        for j in range(1, capacity + 1):
            # 要取的物品重量 大于 当前背包剩余的容量时，就不选了，
            if weights[i - 1] > j:
                # 就取上一个最大的价值
                dp[i][j] = dp[i - 1][j]
                # print(f'就取上一个最大的价值')
            else:
                # 放当前的物品 比较 上一个物品看 价值哪个大， 每一列就是已装容量的最大价值
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1])
        for x in dp:
            print(x)
        print('--' * 10)
    return dp[-1][-1]


weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 8


# 简化方程
def beibao3(weights: list, values: list, capacity: int):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for j in range(capacity, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[-1]


'''
-----------------------------------------------------------------------------
'''


# 简化方程，找到最优解的最大值
def backpack(nums: [], capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(nums)):
        for j in range(capacity, nums[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])

    return dp[capacity]


# 简化方程，找到最优解的组合数
def backpack(nums: [], capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(nums)):
        for j in range(capacity, nums[i] - 1, -1):
            dp[j] += dp[j - nums[i]]
    return dp[capacity]


'''
-----------------------------------------------------------------------------
'''


def backpack(nums: [], capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(nums)):
        for j in range(capacity):
            dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])

    return dp[capacity]

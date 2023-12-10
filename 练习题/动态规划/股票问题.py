#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 22:09
# @Author  : MinisterYU
# @File    : 股票问题.py


prices = [1, 2, 3, 4, 5, 6]

'''
只买卖一次的最大收益
'''


def stock_1(prices):
    '''

    '''

    dp = [[0] * 2 for _ in prices]

    dp[0][0] = 0  # 卖出
    dp[0][1] = -prices[0]  # 买入
    profit = 0
    for i in range(1, len(prices)):
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])  # 可以买卖N多次
        # dp[i][1] = max(dp[i - 1][1], -prices[i])  # 只买卖一次
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        profit = max(dp[i][0], dp[i][1])

    return profit


def stock_1_simple(prices):
    dp = [0, 0]

    dp[0] = 0  # 卖出
    dp[1] = -prices[0]  # 买入

    for i in range(1, len(prices)):
        dp[1] = max(dp[1], dp[0] - prices[i])  # 可以买卖N多次
        # dp[1] = max(dp[1], - prices[i])  # 只买卖一次
        dp[0] = max(dp[0], dp[1] + prices[i])

    return dp[0]


print(stock_1_simple([7, 1, 5, 3, 6, 4]))


def stock_2(prices, fee):
    '''

    '''

    dp = [[0] * 2 for _ in prices]

    dp[0][0] = 0  # 卖出
    dp[0][1] = -prices[0]  # 买入
    profit = 0
    for i in range(1, len(prices)):
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])  # 可以买卖N多次
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
        profit = max(dp[i][0], dp[i][1])

    return profit if profit > 0 else 0


def stock_2_simple(prices, fee):
    dp = [0, 0]

    dp[0] = 0  # 卖出
    dp[1] = -prices[0]  # 买入

    for i in range(1, len(prices)):
        dp[1] = max(dp[1], dp[0] - prices[i])  # 可以买卖N多次
        # dp[1] = max(dp[1], - prices[i])  # 只买卖一次
        dp[0] = max(dp[0], dp[1] + prices[i] - fee)

    return dp[0]


print(stock_2([1, 3, 2, 8, 4, 9], 2))
print(stock_2([9, 8, 7, 1, 2], 3))

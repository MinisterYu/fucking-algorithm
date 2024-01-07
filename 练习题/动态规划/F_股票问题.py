#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 22:09
# @Author  : MinisterYU
# @File    : F_股票问题.py


prices = [1, 2, 3, 4, 5, 6]

'''
只买卖一次的最大收益
'''


class Solution:
    # TODO ------------- 股票买卖 -----------------
    # TODO 121. 买卖股票的最佳时机 -- 买卖一次
    def maxProfit(self, prices):
        dp = [[0] * 2 for _ in range(len(prices) + 1)]
        # 不持有股票的状态
        dp[0][0] = 0
        # 持有股票的状态
        dp[0][1] = -prices[0]

        for i in range(1, len(prices) + 1):
            # 第 i 天持有 :    i-1 持有， 到i天持有
            dp[i][1] = max(dp[i - 1][1], -prices[i - 1])
            # 第 i 天不持有 : i-1 不持有， 到i不持有
            dp[i][0] = max(dp[i - 1][0], dp[i][1] + prices[i - 1])

        for i in dp:
            print(i)
        return max(dp[-1][0], dp[-1][1])

    # TODO 121. 买卖股票的最佳时机 -- 买卖N次
    def maxProfit_Ntimes(self, prices: list) -> int:
        dp = [0, 0]

        dp[0] = 0
        dp[1] = -prices[0]
        for i in range(len(prices)):
            dp[1] = max(dp[1], dp[0] - prices[i])
            dp[0] = max(dp[0], dp[1] + prices[i])
        return max(dp)

    # TODO 714. 买卖股票的最佳时机 -- 买卖N次
    def maxProfit_fee(self, prices: list, fee: int) -> int:
        dp = [0, 0]

        dp[0] = 0
        dp[1] = -prices[0]
        for i in range(len(prices)):
            dp[1] = max(dp[1], dp[0] - prices[i])
            dp[0] = max(dp[0], dp[1] + prices[i] - fee)
        return max(dp)

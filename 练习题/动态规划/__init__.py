#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 22:09
# @Author  : MinisterYU
# @File    : __init__.py.py


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

    # TODO ------------- 背包 -----------------
    # TODO 416 . 01背包(分割数组， 求最大值是否等于目标值）
    def canPartition(self, nums):
        total = sum(nums)
        if total % 2:
            return False
        caps = total // 2
        dp = [0] * (caps + 1)
        for i in range(len(nums)):
            for j in range(caps, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        if dp[-1] == caps:
            return True
        return False

    # TODO 1049 . 01背包(分割数组, 求最大值）
    def lastStoneWeightII(self, stones):
        caps = sum(stones) // 2
        if not caps:
            return 0

        dp = [0] * (caps + 1)
        for i in range(len(stones)):
            for j in range(caps, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])

        return sum(stones) - 2 * dp[-1]

    # TODO 494 . 01背包(求背包装物品的组合数）
    def findTargetSumWays(self, nums, target):

        if (sum(nums) + target) % 2 or sum(nums) + target < 0:
            return 0

        caps = (sum(nums) + target) // 2
        dp = [0] * (caps + 1)

        # 求不同组合方式，在外面初始化1开始计数
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(caps, nums[i] - 1, -1):
                dp[j] = dp[j] + dp[j - nums[i]]

        return dp[-1]

    # TODO 474 . 01背包(背包多属性, 最多装多少个东西）
    def findMaxForm(self, strs, m, n):

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for s in strs:
            count_m = s.count('0')
            count_n = s.count('1')
            for i in range(m, count_m - 1, -1):
                for j in range(n, count_n - 1, -1):
                    # 求最多放多少个物品，在递推这里+1 ： dp[j] , dp[j-nums[i]] + 1
                    dp[i][j] = max(dp[i][j], dp[i - count_m][j - count_n] + 1)

        return dp[-1][-1]

    # TODO 518 . 完全背包（零钱兑换，背包装物品的组合数）
    def change(self, amount, coins):
        dp = [0] * (amount + 1)

        dp[0] = 1
        # 物品在外面是组合，物品在里面是排列
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] = dp[j] + dp[j - coins[i]]
        return dp[j]

    # TODO 377 . 完全背包（排列数，背包装物品的排列数）
    def combinationSum4(self, nums, target):
        dp = [0] * (target + 1)

        dp[0] = 1
        # 物品在外面是组合，物品在里面是排列
        for i in range(1, target + 1):
            for j in range(len(nums)):
                if nums[j] > i:
                    continue
                dp[i] = dp[i] + dp[i - nums[j]]
        return dp[-1]

    # TODO 322 . 完全背包（最少组合数，注意初始化的DP矩阵）
    def coinChange(self, coins, amount):

        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] = min(dp[j], dp[j - coins[i]] + 1)

        return dp[-1] if dp[-1] < amount + 1 else -1

    # TODO 322 . 完全背包（最少组合数，注意初始化的DP矩阵）
    def numSquares(self, n):
        nums = [i * i for i in range(1, n + 1)]
        dp = [n + 1] * (n + 1)
        dp[0] = 0
        for i in range(len(nums)):
            for j in range(nums[i], n + 1):
                dp[j] = min(dp[j], dp[j - nums[i]] + 1)

        return dp[n]

    # TODO 139 . 完全背包 | 拆分单词
    def wordBreak(self, s, wordDict):

        dp = [False] * (len(s) + 1)

        dp[0] = True
        for i in range(1, len(s) + 1):
            for j in range(i):
                if s[j:i] in wordDict and dp[j] == True:
                    dp[i] = True

        return dp[-1]

    # TODO ------------- 打家劫舍 -----------------
    # TODO 198. 打家劫舍: 状态转移，注意初始化，和转移方程
    def rob(self, nums):
        dp = [0] * (len(nums) + 1)
        dp[1] = nums[0]
        for i in range(2, len(nums) + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[-1]

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

    # TODO ------------- 子序列 -----------------

    # TODO 300. 最长子序列 1维
    def lengthOfLIS(self, nums):
        dp = [1] * (len(nums))

        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:  # 判断条件
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

    # TODO 714. 最长连续子序列  1维
    def findLengthOfLCIS(self, nums):

        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:  # 判断条件
                dp[i] = dp[i - 1] + 1

        return max(dp)

    # TODO 718. 最长连续子序列  2维
    def findLength(self, nums1, nums2):

        dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
        ans = 0
        for i in range(1, len(nums1) + 1):
            for j in range(1, len(nums2) + 1):
                if nums1[i - 1] == nums2[j - 1]:  # 判断条件
                    dp[i][j] = dp[i - 1][j - 1] + 1
            ans = max(ans, max(dp[i]))

        ''' dp:
        [0, 0, 0, 0, 0, 0]
        [0, 1, 1, 1, 1, 1]
        [0, 1, 2, 2, 2, 2]
        [0, 1, 2, 3, 3, 3]
        [0, 1, 2, 3, 4, 4]
        [0, 1, 2, 3, 4, 5]
        '''
        return ans

    # TODO 1143. 最长子序列 2维 | "ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
    def longestCommonSubsequence(self, text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:  # 判断条件
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    # TODO 53. 最大子序和 | 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    def maxSubArray(self, nums) -> int:

        dp = [0] * (len(nums))

        dp[0] = nums[0]

        for i in range(1, len(nums)):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

    # TODO 回文串：(最长连续
    def longestPalindrome_连续(self, s):
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = ""

        for i in range(n - 1, -1, -1):  # 从下往上遍历
            for j in range(i, n):  # 从做往右遍历
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    # dp[i][j] 定义 s字符串中，从i到j的最长回文子串
                    # dp[i][j] = s[i] 等于 s[j] 且 （ 左右间隔小于2，（aa,aba） 或者上一个状态为一个回文串 ）
                    dp[i][j] = True
                if dp[i][j] and j - i + 1 > len(res):
                    res = s[i:j + 1]

        return res


    def longestPalindrome_子序列(self, s):
        n = len(s)
        dp = [[0] * n for _ in range(n)]

        for i in range(len(s)): # 初始化
            dp[i][i] = 1

        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])

        return dp[0][n - 1] # 取右上角的那个，因为是从左往右，从下往上在遍历


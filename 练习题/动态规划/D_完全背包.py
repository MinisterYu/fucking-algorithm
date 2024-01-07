# coding:utf-8

class Solution:
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


if __name__ == '__main__':
    pass

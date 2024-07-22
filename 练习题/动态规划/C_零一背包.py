# coding:utf-8
class Solution:
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

    def canPartition2(self, nums):
        n = len(nums)
        s = sum(nums)
        if s % 2:
            return False
        target = s // 2
        dp = [True] + [False] * target
        for i in range(n):
            for j in range(target, nums[i] - 1, -1):
                # 如果 j 可以通过前 i-1 个数得到，那么 dp[i][j] = True。这表示我们可以选择不使用第 i 个数，直接继承前 i-1 个数的选择方案。
                # 如果 j 可以通过前 i-1 个数得到，那么 dp[i][j+nums[i]] = True。这表示我们可以选择使用第 i 个数，并将其加到和为 j 的选择方案中。
                dp[j] = dp[j] or dp[j - nums[i]]
        print(dp)
        return dp[-1]

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


if __name__ == '__main__':
    pass

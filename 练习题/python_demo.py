# coding:utf-8

from typing import List
class Solution:
    def numTeams(self, rating: List[int]) -> int:
        res = 0
        n = len(rating)
        for i in range(1, n - 1):
            left_small, left_big = 0, 0
            right_small, right_big = 0, 0
            for j in range(i):
                if rating[j] < rating[i]:
                    left_small += 1
                else:
                    left_big += 1
            for j in range(i + 1, n):
                if rating[j] < rating[i]:
                    right_small += 1
                else:
                    right_big += 1

            res += left_small * right_big + left_big * right_small
        return res


    def lastStoneWeightII(self, stones: List[int]) -> int:
        cap = sum(stones) // 2
        dp = [0] * (cap + 1)
        for i in range(len(stones)):
            for j in range(cap, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return sum(stones) - 2 * dp[-1]


    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return 0
        dp = [0] * len(nums)
        for i in range(2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i - 1] + 1
        return sum(dp)

    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        n = len(books)
        dp = [0] + [float('inf')] * n
        for i in range(1, n + 1):
            width = 0
            height = 0
            for j in range(i, 0, -1):
                width += books[j - 1][0]
                if width > 0:
                    break
                height = max(height, books[j - 1][1])
                dp[i] = min(dp[i], dp[i - 1] + height)
        return dp[-1]

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        '''
输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]

        '''
        intervals.sort(key=lambda x: x[0])
        n = len(intervals)
        i = 0
        res = []
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i += 1
        res.append(newInterval)
        while 1 < n and newInterval[1] < intervals[i][0]:
            res.append(intervals[i])
            i += 1

        return res






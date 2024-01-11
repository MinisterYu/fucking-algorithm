#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/2 11:23
# @Author  : MinisterYU
# @File    : 一些实践.py
from typing import List


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # https://leetcode.cn/problems/word-search/
        # 二维数组搜索，与岛屿问题类似

        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        dir = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def backtrack(i, j, k):
            # 如果单词的所有字母都匹配完成，返回 True
            if k == len(word):
                return True

            # 如果当前位置超出网格范围，或者当前位置已经访问过，或者当前位置的字母与单词不匹配，返回 False
            if not 0 <= i < m or not 0 <= j < n or not board[i][j] == word[k]:
                return False

            if visited[i][j]:
                return False

            visited[i][j] = True
            res = False
            for x, y in dir:
                res = backtrack(i + x, j + y, k + 1) or res
            visited[i][j] = False

            return res

        for i in range(m):
            for j in range(n):
                if backtrack(i, j, 0):
                    return True

        return False

    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        nums.sort(key=lambda x: -x)  # 分组求和时，逆序排序可以加速算法速度

        total_sum = sum(nums)
        if total_sum % k != 0:
            return False

        target_sum = total_sum // k
        subset_sums = [0] * k

        def backtrack(index):
            if index == len(nums):
                return all(subset_sum == target_sum for subset_sum in subset_sums)

            for i in range(k):
                if i > 0 and subset_sums[i - 1] == subset_sums[i]:
                    continue
                if subset_sums[i] + nums[index] <= target_sum:
                    subset_sums[i] += nums[index]
                    if backtrack(index + 1):
                        return True
                    subset_sums[i] -= nums[index]

            return False

        return backtrack(0)


if __name__ == '__main__':
    so = Solution()
    so.canPartitionKSubsets([4, 3, 2, 3, 5, 2, 1], 4)

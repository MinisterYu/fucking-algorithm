#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 14:45
# @Author  : MinisterYU
# @File    : 二维数组.py
def longestIncreasingPath(matrix):
    # BM61 矩阵最长递增路径
    # https://www.nowcoder.com/practice/7a71a88cdf294ce6bdf54c899be967a2?tpId=295&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Foj
    m, n = len(matrix), len(matrix[0])
    memo = [[0] * n for _ in range(m)]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def dfs(i, j):
        # if memo[i][j] != 0:
        #     return memo[i][j]
        max_len = 1
        for dx, dy in dirs:
            x, y = i + dx, j + dy
            if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] > matrix[i][j]:
                length = dfs(x, y) + 1
                max_len = max(max_len, length)
        # memo[i][j] = max_len
        return max_len

    if not matrix:
        return 0

    max_length = 0
    for i in range(m):
        for j in range(n):
            length = dfs(i, j)
            max_length = max(max_length, length)

    return max_length

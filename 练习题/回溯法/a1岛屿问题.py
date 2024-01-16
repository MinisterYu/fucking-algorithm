#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 20:19
# @Author  : MinisterYU
# @File    : a1岛屿问题.py
from typing import List


def two_dimensional_array(grid: List[List[int]], i: int, j: int, visited: List[List[bool]]):
    '''
    # 二维矩阵遍历框架
    '''
    m, n = len(grid), len(grid[0])
    if not 0 <= i < m or not 0 <= j < n:  # 超出索引边界
        return

    if visited[i][j]:  # 已遍历过 (i, j)
        return
    # 进入节点 (i, j)
    visited[i][j] = True
    two_dimensional_array(grid, i - 1, j, visited)  # 上
    two_dimensional_array(grid, i + 1, j, visited)  # 下
    two_dimensional_array(grid, i, j - 1, visited)  # 左
    two_dimensional_array(grid, i, j + 1, visited)  # 右


'''
这里额外说一个处理二维数组的常用小技巧，
你有时会看到使用「方向数组」来处理上下左右的遍历，和前文 union-find 算法详解 的代码很类似：
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def dfs(grid: List[List[int]], i: int, j: int, visited: List[List[bool]]) -> None:
    m, n = len(grid), len(grid[0])
    if i < 0 or j < 0 or i >= m or j >= n:
        # 超出索引边界
        return
    if visited[i][j]:
        # 已遍历过 (i, j)
        return

    # 进入节点 (i, j)
    visited[i][j] = True
    # 递归遍历上下左右的节点
    for dx, dy in directions:
        dfs(i + dx, j + dy)
    # 离开节点 (i, j)
'''


class Solution:

    def numIslands(self, grid: List[List[str]]) -> int:
        # https://leetcode.cn/problems/number-of-islands/
        '''
        给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

        岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

        此外，你可以假设该网格的四条边均被水包围。
        '''

        # 从 (i, j) 开始，将与之相邻的陆地都变成海水

        m, n = len(grid), len(grid[0])

        def dfs(grid, i, j):
            # 超出索引边界                        # 已经被淹了，直接返回
            if not 0 <= i < m or not 0 <= j < n or grid[i][j] == '0':
                return

            grid[i][j] = '0'  # 访问节点
            # 淹没上下左右的陆地
            dfs(grid, i + 1, j)
            dfs(grid, i - 1, j)
            dfs(grid, i, j + 1)
            dfs(grid, i, j - 1)

        # ----
        count = 0

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1  # 遇到岛屿记录一个
                    dfs(grid, i, j)  ## 然后使用 DFS 将岛屿淹了
        return count

    def closedIsland(self, grid: List[List[int]]) -> int:
        # https://leetcode.cn/problems/number-of-closed-islands/
        '''
        二维矩阵 grid 由 0 （土地）和 1 （水）组成。
        岛是由最大的4个方向连通的 0 组成的群，封闭岛是一个 完全 由1包围（左、上、右、下）的岛。
        请返回 封闭岛屿 的数目。
        '''

    def closedIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        def dfs(grid, i, j):
            if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] == 1:
                return

            grid[i][j] = 1

            dfs(grid, i + 1, j)
            dfs(grid, i - 1, j)
            dfs(grid, i, j + 1)
            dfs(grid, i, j - 1)

        for i in grid:
            print(i)
        print('-')

        for j in range(n):
            # 把靠上边的岛屿淹掉，直到遇到海水
            dfs(grid, 0, j)
            # 把靠下边的岛屿淹掉，直到遇到海水
            dfs(grid, m - 1, j)

        for i in range(m):
            # 把靠左边的岛屿淹掉，直到遇到海水
            dfs(grid, i, 0)
            # 把靠右边的岛屿淹掉，直到遇到海水
            dfs(grid, i, n - 1)
        for i in grid:
            print(i)

        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:  # 中间出现陆地了
                    count += 1
                    dfs(grid, i, j)  # 继续淹旁边
        print('-')
        for i in grid:
            print(i)
        return count

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # https://leetcode.cn/problems/max-area-of-island/
        '''
        给你一个大小为 m x n 的二进制矩阵 grid 。
        岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。
        你可以假设 grid 的四个边缘都被 0（代表水）包围着。
        岛屿的面积是岛上值为 1 的单元格的数目。
        计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0
        '''
        # 记录岛屿的最大面积
        res = 0
        m, n = len(grid), len(grid[0])

        def dfs(i: int, j: int) -> int:
            if not 0 <= i < m or not 0 <= j < n or grid[i][j] == 0:
                return 0
            # 将 (i, j) 变成海水
            grid[i][j] = 0
            # 淹没与 (i, j) 相邻的陆地，并返回淹没的陆地面积
            return dfs(i + 1, j) + dfs(i, j + 1) + dfs(i - 1, j) + dfs(i, j - 1) + 1

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # 淹没岛屿，并更新最大岛屿面积
                    res = max(res, dfs(i, j))
        return res

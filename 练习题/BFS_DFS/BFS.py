#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 22:49
# @Author  : MinisterYU
# @File    : __init__.py.py
# 注意：python 代码由 chatGPT🤖 根据我的 java 代码翻译，旨在帮助不同背景的读者理解算法逻辑。
# 本代码不保证正确性，仅供参考。如有疑惑，可以参照我写的 java 代码对比查看。
import math
from typing import List, Set
from collections import deque


class Node:
    def __init__(self, val: int):
        self.val = val
        self.neighbors = []


def BFS(start: Node, target: Node) -> int:
    q = deque()  # 核心数据结构
    visited = set()  # 避免走回头路
    q.append(start)  # 将起点加入队列
    visited.add(start)

    step = 0  # 记录扩散的步数

    while q:
        step += 1
        size = len(q)
        # 将当前队列中的所有节点向四周扩散
        for i in range(size):
            cur = q.popleft()
            # 划重点：这里判断是否到达终点
            if cur == target:
                return step
            # 将cur相邻节点加入队列
            for x in cur.neighbors:
                if x not in visited:
                    q.append(x)
                    visited.add(x)
    # 如果走到这里，说明在图中没有找到目标节点


class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        # https://leetcode.cn/problems/rotting-oranges/?envType=study-plan-v2&envId=top-100-liked
        # todo 腐烂的橘子
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j))
                elif grid[i][j] == 1:
                    fresh += 1

        if fresh == 0:
            return 0

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        minutes = 0
        while queue:
            for _ in range(len(queue)):
                x, y = queue.popleft()
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                        grid[nx][ny] = 2
                        fresh -= 1
                        queue.append((nx, ny))
            minutes += 1
        return minutes - 1 if fresh == 0 else -1



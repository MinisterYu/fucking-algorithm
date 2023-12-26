#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/18 12:08
# @Author  : MinisterYU
# @File    : z3活字印花1079，全排列，求排列总数.py
'''
你有一套活字字模 tiles，其中每个字模上都刻有一个字母 tiles[i]。返回你可以印出的非空字母序列的数目。
注意：本题中，每个活字字模只能使用一次。

示例 1：
输入："AAB"
输出：8
解释：可能的序列为 "A", "B", "AA", "AB", "BA", "AAB", "ABA", "BAA"。
示例 2：

输入："AAABBC"
输出：188
示例 3：

输入："V"
输出：1
'''


class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        self.count = 0
        visited = [False] * len(tiles)
        self.backtrack(tiles, visited)
        return self.count

    def backtrack(self, tiles, visited):
        for i in range(len(tiles)):
            if visited[i]:
                continue
            if i > 0 and tiles[i] == tiles[i - 1] and not visited[i - 1]:
                continue
            visited[i] = True
            self.count += 1
            self.backtrack(tiles, visited)
            visited[i] = False


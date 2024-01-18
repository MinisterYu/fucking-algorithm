#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 10:49
# @Author  : MinisterYU
# @File    : DFS.py
from typing import List
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = set()
        n = len(rooms)
        self.num = 0

        def dfs(index):
            visited.add(index) # 进入N号房
            self.num += 1
            for i in rooms[index]: # 看房间里面的钥匙
                if i not in visited:
                    dfs(i) # 遍历进入房间

        dfs(0)
        if self.num == n: # 如果房间都进入过了，则返回true
            return True
        return False

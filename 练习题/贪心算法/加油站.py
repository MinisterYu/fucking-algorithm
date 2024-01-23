#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 20:49
# @Author  : MinisterYU
# @File    : 加油站.py
from typing import List
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        '''
        假设我们从第 i 个加油站开始，我们需要维护两个变量：total_gas 表示当前汽车的总汽油量，current_gas 表示当前加油站到下一个加油站的汽油量。

        我们可以从第 i 个加油站出发，将 total_gas 初始化为 gas[i]，然后遍历每个加油站。在每个加油站上，我们首先检查当前汽车的汽油量是否足够到达下一个加油站，即 current_gas = gas[i] - cost[i]。如果足够，我们可以继续前进到下一个加油站，并将 total_gas 更新为 total_gas + current_gas。否则，我们无法到达下一个加油站，需要重新选择起点。

        如果在遍历完所有加油站后，total_gas 的值大于等于 0，说明我们可以按顺序绕环路行驶一周，返回起点的编号。否则，返回 -1。
        '''
        n = len(gas)
        total_gas = 0
        current_gas = 0
        start_station = 0

        for i in range(n):
            total_gas = total_gas + gas[i] - cost[i]
            current_gas = current_gas + gas[i] - cost[i]

            if current_gas < 0:
                start_station = i + 1
                current_gas = 0

        if total_gas >= 0:
            return start_station
        else:
            return -1
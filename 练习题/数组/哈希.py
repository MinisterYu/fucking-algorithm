#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 11:17
# @Author  : MinisterYU
# @File    : 哈希.py
import heapq
from typing import List


def findMissingNumber(nums):
    # 给定一个长度为 n 的数组，其中的值是从 0 到 n 的范围内，且只有一个数字缺失，我们可以使用原地哈希的方法来找出缺失的数字。

    '''
    在这个示例中，我们首先遍历数组 nums，将出现过的数字进行标记。对于每个数字 nums[i]，
    我们将其对应的索引 index 处的数字取负数，表示该数字已经出现过。
    然后，我们再次遍历数组，找到第一个未被标记的数字，即为缺失的数字。
    如果没有找到未被标记的数字，则说明缺失的是 n。
    通过这种原地哈希的方式，我们可以在不使用额外空间的情况下找出缺失的数字。
    '''
    n = len(nums)
    for i in range(n):
        if nums[i] == 0:
            nums[i] = n + 1

    # 标记出现过的数字
    for i in range(n):
        index = abs(nums[i])
        if index <= n:
            nums[index - 1] = -abs(nums[index - 1])

    # 找到未被标记的数字
    for i in range(n):
        if nums[i] > 0:
            return i + 1

    # 如果没有未被标记的数字，则缺失的是 n
    return n + 1


def findMissingPositive(nums):
    # 找出未排序的整数数组中没有出现的最小的正整数，可以使用原地哈希的方法。
    '''
    我们首先将所有非正整数置为 n+1，这样它们不会影响到最小的正整数的查找。
    然后，我们遍历数组，将出现过的正整数进行标记。
    对于每个正整数 num，我们将其对应的索引 num - 1 处的数字取负数，表示该正整数已经出现过。
    最后，我们再次遍历数组，找到第一个未被标记的正整数，即为没有出现的最小的正整数。
    如果都被标记了，则说明没有出现的最小的正整数是 n+1。
    通过这种原地哈希的方式，我们可以在不使用额外空间的情况下找出未排序的整数数组中没有出现的最小的正整数。
    '''
    n = len(nums)

    # 将所有非正整数置为 n+1
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1

    # 标记出现过的正整数
    for i in range(n):
        index = abs(nums[i])
        if index <= n:
            nums[index - 1] = -abs(nums[index - 1])

    # 找到第一个未被标记的正整数
    for i in range(n):
        if nums[i] > 0:
            return i + 1

    # 如果都被标记了，则返回 n+1
    return n + 1


def maxOperations(nums: List[int], k: int) -> int:
    # https://leetcode.cn/problems/max-number-of-k-sum-pairs/?envType=study-plan-v2&envId=leetcode-75
    # TODO 给你一个整数数组 nums 和一个整数 k 。每一步操作中，你需要从数组中选出和为 k 的两个整数，并将它们移出数组。
    bucket = {}
    time = 0
    for num in nums:
        if bucket.get(k - num, 0):
            bucket[k - num] -= 1
            time += 1
        else:
            bucket[num] = bucket.get(num, 0) + 1
    return time


def equalPairs(grid: List[List[int]]) -> int:
    n = len(grid)
    counter_map = {}
    res = 0
    # 行遍历
    for i in range(n):
        r_key = ''
        for j in range(n):
            r_key += str(grid[i][j])
        counter_map[r_key] = counter_map.get(r_key, 0) + 1

    for i in range(n):
        c_key = ''
        for j in range(n):
            c_key += str(grid[j][i])
        if c_key in counter_map:
            res += counter_map[c_key]
    return res


equalPairs([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def findMinDifference(timePoints: List[str]) -> int:
    minutes = []
    for time in timePoints:
        hour, minute = map(int, time.split(':'))
        total_minutes = hour * 60 + minute
        minutes.append(total_minutes)

    minutes.sort()

    min_diff = float('inf')
    for i in range(1, len(minutes)):
        diff = minutes[i] - minutes[i - 1]
        min_diff = min(min_diff, diff)

    # 考虑首尾时间的差值
    diff = minutes[0] + (24 * 60 - minutes[-1])
    min_diff = min(min_diff, diff)

    return min_diff


def containsNearbyAlmostDuplicate(nums: List[int], k: int, t: int) -> bool:
    # abs(nums[i] - nums[j]) <= t
    # abs(i - j) <= k
    if not nums:
        return False
    if k == 0:
        return True

    def lower_bound(target, lists):
        left = 0
        right = len(lists)
        while left < right:
            mid = (right - left) // 2 + left
            if lists[mid] < target:
                left = mid + 1
            else:
                right = mid

        return left

    heap = []
    for i in range(len(nums)):
        lower = lower_bound(nums[i] - t, heap)
        if lower != len(heap) and heap[lower] <= nums[i] + t:
            return True

        if nums[i] not in heap:
            heapq.heappush(heap, nums[i])

        if i >= k:
            heap.remove(nums[i - k])

    return False


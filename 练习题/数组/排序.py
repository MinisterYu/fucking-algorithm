#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/1 13:07
# @Author  : MinisterYU
# @File    : 排序.py
import heapq
from typing import List
from collections import Counter


def containsNearbyAlmostDuplicate(nums: List[int], indexDiff: int, valueDiff: int
                                  ) -> bool:
    if valueDiff < 0:
        return False

    buckets = {}  # 桶
    bucket_size = valueDiff + 1  # 一个桶的大小

    for i in range(len(nums)):

        bucket_no = nums[i] // bucket_size  # 确定放进桶的位置

        if bucket_no in buckets:  # 如果区间桶中已经有元素了，则说明找到了
            return True

        buckets[bucket_no] = nums[i]

        # left_bucket_no = bucket_no - 1
        # if left_bucket_no in buckets and abs(buckets[left_bucket_no] - nums[i]) <= valueDiff:
        #     return True

        # right_bucket_no = bucket_no + 1
        # if right_bucket_no in buckets and abs(buckets[right_bucket_no] - nums[i]) <= valueDiff:
        #     return True

        if i >= indexDiff:
            buckets.pop(nums[i - indexDiff] // bucket_size)

    return False

#
# print(containsNearbyAlmostDuplicate(nums=[8, 7, 15, 1, 6, 1, 9, 15], indexDiff=1, valueDiff=3))
if __name__ == '__main__':
    import random
    heap = []
    visited = set()
    for i in range(20):
        value = random.randint(0, 100)
        if value not in visited:
            heapq.heappush(heap, value)
            visited.add(value)
    print(heap)
    while heap:
        print(heapq.heappop(heap))


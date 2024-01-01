#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 13:07
# @Author  : MinisterYU
# @File    : 桶排序.py
# from sortedcontainers import SortedList
import heapq
from typing import List


class Solution:

    # 没看明白。。。
    # todo 220. 存在重复元素 III
    # https://leetcode.cn/problems/contains-duplicate-iii/
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        if valueDiff < 0 or indexDiff < 0:
            return False

        buckets = {}
        buckets_size = valueDiff + 1  # 桶的大小设成t+1更加方便

        for i in range(len(nums)):
            bucket_num = nums[i] // buckets_size  # 放入哪个桶

            if bucket_num in buckets:  # 桶中已经有元素了
                return True

            buckets[bucket_num] = nums[i]  # 把nums[i]放入桶中

            if (bucket_num - 1) in buckets and abs(buckets[bucket_num - 1] - nums[i]) <= valueDiff:  # 检查前一个桶
                return True

            if (bucket_num + 1) in buckets and abs(buckets[bucket_num + 1] - nums[i]) <= valueDiff:  # 检查后一个桶
                return True

            # 如果不构成返回条件，那么当i >= k 的时候就要删除旧桶了，以维持桶中的元素索引跟下一个i+1索引只差不超过k
            if i >= indexDiff:
                buckets.pop(nums[i - indexDiff] // buckets_size)

        return False


if __name__ == '__main__':
    so = Solution()
    test1 = so.containsNearbyAlmostDuplicate([8, 2, 9, 1, 5, 9], 2, 1)
    print(test1)

    m = [(6, 6), (2, 2), (3, 3), (4, 4)]
    heap = []
    for i, v in m:
        heapq.heappush(heap, (v, i))
    print(heap)

    m = [6, 2, 3, 4]
    heap2 = []
    for i in m:
        heapq.heappush(heap2, i)
    print(heap2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 13:07
# @Author  : MinisterYU
# @File    : 桶排序.py
# from sortedcontainers import SortedList
from typing import List


class Solution:

    # todo 220. 存在重复元素 III
    # https://leetcode.cn/problems/contains-duplicate-iii/
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        if valueDiff < 0 or indexDiff < 0:
            return False

        all_buckets = {}
        bucket_size = valueDiff + 1  # 桶的大小设成t+1更加方便

        for i in range(len(nums)):
            bucket_num = nums[i] // bucket_size  # 放入哪个桶

            if bucket_num in all_buckets:  # 桶中已经有元素了
                return True

            all_buckets[bucket_num] = nums[i]  # 把nums[i]放入桶中

            if (bucket_num - 1) in all_buckets and abs(all_buckets[bucket_num - 1] - nums[i]) <= valueDiff:  # 检查前一个桶
                return True

            if (bucket_num + 1) in all_buckets and abs(all_buckets[bucket_num + 1] - nums[i]) <= valueDiff:  # 检查后一个桶
                return True

            # 如果不构成返回条件，那么当i >= k 的时候就要删除旧桶了，以维持桶中的元素索引跟下一个i+1索引只差不超过k
            if i >= indexDiff:
                all_buckets.pop(nums[i - indexDiff] // bucket_size)

        return False


if __name__ == '__main__':
    so = Solution()
    test1 = so.containsNearbyAlmostDuplicate([1, 5, 9, 1, 5, 9], 2, 3)

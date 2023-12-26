#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 14:48
# @Author  : MinisterYU
# @File    : z1划分为k个相等的子集698.py
import functools
from typing import List
import time


# https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/description/
# 给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。

class Solution:

    # 以数字的视角解决
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        total = sum(nums)
        # 排除一些基本情况
        if k > n:
            return False
        if k == n and k // n != nums[0]:
            return False
        if total % k:
            return False
        # 所有分出来的桶需要装 target 个数字
        target = total // k
        # k 个桶
        bucket = [0] * k
        # 倒序排列，更好剪枝
        nums.sort(reverse=True)

        def backtrack(index, nums, bucket, target):
            # 所有数都填完了，每个桶里的数字和都是 target
            if index == len(nums):
                for i in range(len(bucket)):
                    if bucket[i] != target:
                        return False
                return True
            # 每个桶都尝试一下
            for i in range(len(bucket)):
                # 如果加进去这个数，这个桶的数字和就超过了 target，那就不能加了
                if bucket[i] + nums[index] > target:
                    continue
                # 加进去
                bucket[i] += nums[index]
                # 如果这个加法是可行方案，就继续递归下去
                if backtrack(index + 1, nums, bucket, target):
                    return True
                # 加完又要撤消加法，恢复现场，继续尝试别的加法
                bucket[i] -= nums[index]

            # 无解，返回false
            return False

        return backtrack(0, nums, bucket, target)

    # 以桶的视角解决
    def canPartitionKSubsets_from_bukect(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        total = sum(nums)
        # 排除一些基本情况
        if k > n:
            return False
        if k == n and k // n != nums[0]:
            return False
        if total % k:
            return False
        # 所有分出来的桶需要装 target 个数字
        target = total // k

        used = [False] * len(nums)

        def backtrack(k, bucket, nums, start, used, target):
            if k == 0:
                # 所有桶都被装满了，而且 nums 一定全部用完了
                # 因为 target == sum / k
                return True
            if bucket == target:
                # 装满了当前桶，递归穷举下一个桶的选择
                # 让下一个桶从 nums[0] 开始选数字
                return backtrack(k - 1, 0, nums, 0, used, target)

            # 从 start 开始向后探查有效的 nums[i] 装入当前桶
            for i in range(start, len(nums)):

                # 如果数字已经被加入桶了，则不能再选择
                if used[i]:
                    continue

                # 当前桶装不下 nums[i]
                if nums[i] + bucket > target:
                    continue

                # 标记选择数字i，并加入到桶中
                used[i] = True
                bucket += nums[i]

                # 如果此方案可行，则继续递归分解
                if backtrack(k, bucket, nums, i + 1, used, target):
                    return True

                # 撤销选择
                used[i] = False
                bucket -= nums[i]

            return False

        return backtrack(k, 0, nums, 0, used, target)


if __name__ == '__main__':
    so = Solution()
    bg = time.time()
    # res = so.canPartitionKSubsets([3, 9, 4, 5, 8, 8, 7, 9, 3, 6, 2, 10, 10, 4, 10, 2], 10)
    res = so.canPartitionKSubsets_from_bukect([3, 9, 4, 5, 8, 8, 7, 9, 3, 6, 2, 10, 10, 4, 10, 2], 10)
    ed = time.time()
    print(f'res {res} cost {ed - bg}')

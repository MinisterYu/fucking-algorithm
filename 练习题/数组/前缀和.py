#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/2 16:33
# @Author  : MinisterYU
# @File    : 前缀和.py
from typing import List
from itertools import accumulate


class Solution:

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # https://leetcode.cn/problems/minimum-size-subarray-sum/
        res = len(nums) + 1
        preSum = [0] + list(accumulate(nums))

        left = 0  # 左指针
        for right in range(1, len(preSum)):
            while preSum[right] - preSum[left] >= target:
                res = min(res, right - left)
                left += 1
        return res if res < len(nums) + 1 else 0

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # https://leetcode.cn/problems/product-of-array-except-self/
        n = len(nums)
        left = [1] * n
        right = [1] * n

        for i in range(1, n):
            left[i] = left[i - 1] * nums[i - 1]

        for i in range(n - 2, -1, -1):
            right[i] = right[i + 1] * nums[i + 1]

        ans = [left[i] * right[i] for i in range(n)]
        return ans

    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        # https://leetcode.cn/problems/continuous-subarray-sum/description/
        if len(nums) < 2:
            return False

        hashmap = {0: -1}
        premod = 0
        for i in range(len(nums)):
            premod = (premod + nums[i]) % k

            if premod not in hashmap:
                hashmap[premod] = i
            else:
                if i - hashmap[premod] >= 2:
                    return True
        return False

    def findMaxLength(self, nums: List[int]) -> int:
        # https://leetcode.cn/problems/contiguous-array/
        '''
        创建一个字典 prefix_sum，初始值为 {0: -1}，用于记录当前的前缀和以及对应的索引。
        初始化变量 max_length 为 0，用于记录最长连续子数组的长度。
        初始化变量 count 为 0，用于记录当前的前缀和。
        遍历二进制数组 nums，对于每个元素 num：
        如果 num 为 0，则将 count 减 1。
        如果 num 为 1，则将 count 加 1。
        如果 count 在 prefix_sum 中存在，则更新 max_length 为当前索引与 prefix_sum[count] 的差值的最大值。
        如果 count 在 prefix_sum 中不存在，则将 count 添加到 prefix_sum 中，并将其对应的索引设置为当前索引。
        返回 max_length。
        '''
        prefix_sum = {0: -1}
        max_length = 0
        count = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                count += 1
            else:
                count -= 1
            if count not in prefix_sum:
                prefix_sum[count] = i
            else:
                if count in prefix_sum and count == 0:
                    max_length = max(max_length, i - prefix_sum[count])
        return max_length

    def subarraySum(self, nums: List[int], k: int) -> int:
        # https://leetcode.cn/problems/subarray-sum-equals-k/
        '''
        创建一个字典 prefix_sum，初始值为 {0: 1}，用于记录当前的前缀和以及对应的出现次数。
        初始化变量 count 为 0，用于记录和为 k 的子数组的个数。
        初始化变量 prefix 为 0，用于记录当前的前缀和。
        遍历整数数组 nums，对于每个元素 num：
        将 prefix 加上 num，得到当前的前缀和。
        计算 prefix - k，得到需要查找的目标前缀和。
        如果目标前缀和在 prefix_sum 中存在，则将其对应的出现次数加到 count 中。
        将当前前缀和的出现次数加 1，更新到 prefix_sum 中。
        返回 count。
        '''
        prefix_sum = {0: 1}
        count = 0
        prefix = 0

        for num in nums:
            prefix += num
            target = prefix - k

            if target in prefix_sum:
                count += prefix_sum[target]
            prefix_sum[prefix] = prefix_sum.get(prefix, 0) + 1
        print(prefix_sum)
        print(count)
        return count


if __name__ == '__main__':
    so = Solution()
    so.subarraySum([4, 2, 1], 3)

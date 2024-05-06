#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/2 16:33
# @Author  : MinisterYU
# @File    : 前缀和.py
import itertools
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
        # 除自身以外的乘积
        prefix = [1] * len(nums)
        for i in range(1, len(nums)):
            prefix[i] = prefix[i - 1] * nums[i - 1]

        post = [1] * len(nums)
        for i in range(len(nums) - 2, -1, -1):
            post[i] = post[i + 1] * nums[i + 1]

        res = []
        for i in range(len(nums)):
            res.append(prefix[i] * post[i])
        return res

    def pivotIndex(self, nums: List[int]) -> int:
        # https://leetcode.cn/problems/find-pivot-index/?envType=study-plan-v2&envId=leetcode-75
        # 724. 寻找数组的中心下标
        n = len(nums)
        prefix = [0] * n

        for i in range(1, n):
            prefix[i] = prefix[i - 1] + nums[i - 1]

        post = [0] * n
        for i in range(n - 2, -1, -1):
            post[i] = post[i + 1] + nums[i + 1]

        for i in range(n):
            if prefix[i] == post[i]:
                return i
        return -1

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
            if count in prefix_sum:
                max_length = max(max_length, i - prefix_sum[count])
            else:
                prefix_sum[count] = i
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

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        # https://leetcode.cn/problems/car-pooling/
        passengers = [0] * (max(i[2] for i in trips) + 1)

        for trip in trips:
            passengers_count = trip[0]
            start = trip[1]
            end = trip[2]

            passengers[start] += passengers_count  # 有顾客上车了
            passengers[end] -= passengers_count  # 有顾客下车了

        total_passengers = 0
        for count in passengers:
            total_passengers += count
            if total_passengers > capacity:
                return False

        return True

    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        count = 0
        stops = {}
        for index, char in enumerate(garbage):
            count += len(char)
            for c in char:
                stops[c] = index

        prefix = list(itertools.accumulate(travel))
        print(prefix, stops)
        for i in stops.values():
            if i - 1 >= 0:
                count += prefix[i - 1]
        return count

    def maxSum(self, grid: List[List[int]]) -> int:
        '''
        因为数组是有序的, 对数任意一个数nums[i],
        在数nums[i]左边的数比nums[i]小, 在nums[i]右边的数比nums[i]大,
        因此, 计算nums[i]和其他数的差绝对值之和可以分割为两部分来进行计算.
        首先计算前缀和数组prefixSum[], prefixSum[i]表示前i个数之和.
        对于nums[i]的左半部分, nums[i]与其他数的差绝对值之和可计算为:

        sumOfLeftDifferences = (i+1)*nums[i]-prefixSum[i];
        对于nums[i]的右半部分, nums[i]与其他数的绝对值之和可计算为:
        sumOfRightDifferences = prefixSum[nums.length-1]-prefixSum[i]-nums[i]*(nums.length-1-i);
        所以, 当前nums[i]与左右其他数的绝对值之和为:
        sumOfDifferences = sumOfLeftDifferences+sumOfRightDifferences;
        '''
        pass

    def largestAltitude(self, gain: List[int]) -> int:
        # https://leetcode.cn/problems/find-the-highest-altitude/?envType=study-plan-v2&envId=leetcode-75
        # 1732. 找到最高海拔
        # res = total = 0
        # for i in gain:
        #     total += i
        #     res = max(res, total)
        # return res

        res = [0] * (len(gain) + 1)
        for i in range(1, len(gain) + 1):
            res[i] = res[i - 1] + gain[i - 1]
        return max(res)

    def shortestSeq(self, big: List[int], small: List[int]) -> List[int]:
        n = need = len(small)
        counter = Counter(small)
        res = len(big) + 1
        left = 0
        res_list = [0, 0]
        for right in range(len(big)):
            number = big[right]
            if number in counter:
                if counter[number] > 0 :
                    need -= 1
                counter[number] -= 1

            while need == 0:
                if res > right - left + 1:
                    res = min(res, right - left + 1)
                    res_list = [left, right + 1]
                number = big[left]
                if number in counter:
                    if counter[number] >= 0:
                        need += 1
                    counter[number] += 1
                left += 1
        return res_list


if __name__ == '__main__':
    so = Solution()
    # so.subarraySum([4, 2, 1], 3)
    # nums = [[2, 1, 5], [3, 3, 7]]
    # print(max( i[2] for i in nums  ))
    # res = so.garbageCollection(garbage=["G", "M"], travel=[1])

    # nums1 = [1, 2, 3, 4, 5]
    # nums2 = [3, 4, 5, 6, 7]
    # print(set(nums1) | set(nums2))
    # print(set(nums1) & set(nums2))
    # print(set(nums1) ^ set(nums2))
    # print("   blue is sky the    ".split())
    print(so.findMaxLength([0, 0, 0, 1, 1, 0, 0, 1, 1]))

# coding:utf-8
from typing import List


# TODO 子集， 去重、不去重
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = []
        path = []

        def dfs(start):
            ans.append(path[:])
            if start == len(nums):
                return
            for i in range(start, len(nums)):
                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return ans

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()

        ans = []
        path = []

        def dfs(start):
            ans.append(path[:])
            if start == len(nums):
                return
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return ans


if __name__ == '__main__':
    pass

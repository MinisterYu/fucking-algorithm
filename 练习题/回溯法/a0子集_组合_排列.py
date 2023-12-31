# coding:utf-8
from typing import List


class Solution:
    '''
    形式一、【元素无重、不可复选】，即 nums 中的元素都是唯一的，每个元素最多只能被使用一次，这也是最基本的形式。

            以组合为例，如果输入 nums = [2,3,6,7]，和为 7 的组合应该只有 [7]。


    形式二、【元素可重不可复选】，即 nums 中的元素可以存在重复，每个元素最多只能被使用一次。

            以组合为例，如果输入 nums = [2,5,2,1,2]，和为 7 的组合应该有两种 [2,2,2,1] 和 [5,2]。


    形式三、【元素无重可复选】，即 nums 中的元素都是唯一的，每个元素可以被使用若干次

            以组合为例，如果输入 nums = [2,3,6,7]，和为 7 的组合应该有两种 [2,2,3] 和 [7]
    '''

    def 组合_元素无重复_不可重复选择(self, n: int, k: int):
        # 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
        # https://leetcode.cn/problems/combinations/
        ans = []
        path = []

        def backtrack(index, n, k, ans, path):
            # 结束条件，path搜集长度 == k
            if len(path) == k:
                ans.append(path[:])
                return

            for i in range(index, n + 1):
                path.append(i)
                backtrack(i + 1, n, k, ans, path)
                path.pop()

        backtrack(1, n, k, ans, path)
        return ans

    def 子集_元素无重复_不可重复选择(self, nums):
        # todo 组合、子集： 元素无重复 不可重复选择
        # 给你输入一个 无重复元素 的数组 nums，其中每个元素最多使用一次，请你返回 nums 的所有子集。
        # https://leetcode.cn/problems/subsets/
        path = []
        ans = []

        def backtrack(index: int, nums: List, path: List, ans: List):
            ans.append(path[:])
            for i in range(index, len(nums)):
                path.append(nums[i])
                backtrack(i + 1, nums, path, ans)
                path.pop()

        backtrack(0, nums, path, ans)
        return ans

    def 子集_组合_元素有重复_不可重复选择(self, nums):
        # todo 组合、子集： 元素有重复 不可重复选择
        # 给你一个整数数组 nums，其中可能包含重复元素，请你返回该数组所有可能的子集。
        # https://leetcode.cn/problems/subsets-ii/
        nums.sort()  # 去重必须先对数组排序
        ans = []
        path = []

        def backtrack(index, nums, ans, path):
            ans.append(path[:])
            for i in range(index, len(nums)):
                # 去重：当前层级，元素相同的节点，只选择前一个
                if i > index and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                backtrack(i + 1, nums, ans, path)
                path.pop()

        backtrack(0, nums, ans, path)
        return ans

    def 子集_组合_元素无重复_可重复选择(self, candidates, target):
        # todo 组合、子集： 元素无重复 可以重复选择
        # 给你一个无重复元素的整数数组 candidates 和一个目标和 target，找出 candidates 中可以使数字和为目标数 target 的所有组合。
        # candidates 中的每个数字可以无限制重复被选取。
        # https://leetcode.cn/problems/combination-sum/
        ans = []
        path = []

        def backtrack(index, candidates, target, ans, path):
            if target == 0:
                ans.append(path[:])
                return
            if target < 0:
                return

            for i in range(index, len(candidates)):
                path.append(candidates[i])
                backtrack(i, candidates, target - candidates[i], ans, path)  # i 不加1就是可以重复选择元素
                path.pop()

        backtrack(0, candidates, target, ans, path)
        return ans

        pass

    def 排列_元素无重复_不可重复选择(self, nums):
        # todo 排列： 元素无重复 不重复选择
        # 给定一个不含重复数字的数组 nums，返回其所有可能的全排列。
        # https://leetcode.cn/problems/permutations/

        ans = []
        path = []
        used = [False] * len(nums)

        def backtrack(used, nums, ans, path):
            if len(path) == len(nums):
                ans.append(path[:])
                return

            for i in range(len(nums)):
                if used[i] == True:
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack(used, nums, ans, path)
                path.pop()
                used[i] = False

        backtrack(used, nums, ans, path)
        return ans

    def 排列_元素有重复_不可重复选择(self, nums):
        # todo 排列： 元素有重复 不重复选择
        # 给你输入一个可包含重复数字的序列 nums，请你写一个算法，返回所有可能的全排列
        # https://leetcode.cn/problems/permutations-ii/
        nums.sort()  # 所有去重的，都必须先排序
        ans = []
        path = []
        used = [False] * len(nums)

        def backtrack(used, nums, ans, path):
            if len(nums) == len(path):
                ans.append(path[:])
                return

            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                    # 如果前面的相邻且相等元素没有用过，则跳过 （ not used[i-1] 树层剪掉枝叶 ）
                    # 要选 a 就要选 a` 否则两个都不选， 这样就避免了 [a , a`] 和 [a`, a] 重复的问题
                    continue
                if used[i]:
                    continue

                used[i] = True
                path.append(nums[i])
                backtrack(used, nums, ans, path)
                path.pop()
                used[i] = False

        backtrack(used, nums, ans, path)
        return ans

    def 排列_元素无重复_可重复选择(self, nums):
        # todo 排列： 元素无重复 可重复选择
        # 无题目，used 不去重就是全排列可重复选择
        ans = []
        path = []

        def backtrack(nums, ans, path):
            if len(path) == len(nums):
                ans.append(path[:])
                return

            for i in range(len(nums)):
                path.append(nums[i])
                backtrack(nums, ans, path)
                path.pop()

        backtrack(nums, ans, path)
        return ans

        pass

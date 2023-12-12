#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/5 14:17
# @Author  : MinisterYU
# @File    : __init__.py.py


# TODO 17. 电话号码的字母组合

mapping = ["", "", "abc", "edf", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]


class Solution(object):
    def letterCombinations(self, digits):
        n = len(digits)
        if not n:
            return []
        ans = []
        path = []

        def dfs(i):
            if i == n:
                return ans.append("".join(path))
            for c in mapping[int(digits[i])]:
                path.append(c)
                dfs(i + 1)
                path.pop()

        dfs(0)
        return ans

    # TODO 40. 组合总和 输入：candidates = [2,3,6,7], target = 7  输出：[[2,2,3],[7]]
    def combinationSum2(self, candidates, target):
        candidates.sort()
        ans = []
        path = []

        def dfs(i):
            if sum(path) > target:
                return
            if sum(path) == target:
                ans.append(path[:])

            for j in range(i, len(candidates)):
                if j > i and candidates[j] == candidates[j - 1]:
                    continue
                path.append(candidates[j])
                dfs(j + 1)
                path.pop()

        dfs(0)
        return ans

    # TODO 子集回溯 (不包含重复元素） 给你一个整数数组 nums ，数组中的元素 互不相同 。解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
    def subsets(self, nums):
        ans = []
        path = []
        n = len(nums)

        def dfs(i):
            # 写在外面说明每一行都是答案
            ans.append(path[:])
            if i == n:
                return

            for j in range(i, n):
                path.append(nums[j])
                dfs(j + 1)
                path.pop()

        dfs(0)
        return ans

        # TODO 子集回溯(包含重复元素）
        def subsetsWithDup(nums):
            res = []  # 存储所有子集的结果
            path = []
            nums.sort()  # 对数组进行排序，以便处理重复元素

            def backtrack(start):
                res.append(path[:])  # 将当前路径加入结果

                for i in range(start, len(nums)):
                    # 排序后就可以去重了
                    if i > start and nums[i] == nums[i - 1]:  # 处理重复元素，跳过重复的数字
                        continue
                    path.append(nums[i])  # 将当前元素加入路径
                    backtrack(i + 1)  # 递归进入下一层，注意起始位置为i+1
                    path.pop()  # 回溯，将当前元素从路径中移除

            backtrack(0)  # 从索引0开始回溯，初始路径为空列表
            return res

        # TODO 子集回溯。（切割）分割回文串 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案

    def partition(self, s):
        ans = []
        path = []
        n = len(s)

        def dfs(i):
            if i == n:
                ans.append(path[:])
                return
            for j in range(i, n):
                if s[i:j + 1] == s[i:j + 1][::-1]:
                    path.append(s[i:j + 1])
                    dfs(j + 1)
                    path.pop()

        dfs(0)
        return ans

    # TODO 组合回溯：给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
    def combine(self, n, k):
        ans = []
        path = []

        def dfs(i):
            # 倒序剪枝， 当剩余的数量 小于 目标数量 - 已选数量时，返回
            if i < k - len(path):
                return

            if len(path) == k:
                ans.append(path[:])
                return

            for j in range(i, 0, -1):
                path.append(j)
                dfs(j - 1)
                path.pop()

        dfs(n)
        return ans

    # TODO 组合回溯：找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        path = []
        ans = []

        def dfs(i):
            if sum(path) > n:
                return

            if len(path) == k and sum(path) == n:
                ans.append(path[:])
                return

            for j in range(i, 10):
                path.append(j)
                dfs(j + 1)
                path.pop()

        dfs(1)
        return ans

    # TODO 组合回溯：数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
    def generateParenthesis(self, n):
        res = []  # 存储所有有效的括号组合

        def backtrack(s, left, right):
            if len(s) == 2 * n:  # 当字符串长度达到2n时，表示已经生成了一个有效的括号组合
                res.append(s)
                return

            if left < n:  # 左括号数量小于n时，可以添加左括号
                backtrack(s + '(', left + 1, right)

            if right < left:  # 右括号数量小于左括号数量时，可以添加右括号
                backtrack(s + ')', left, right + 1)

        backtrack('', 0, 0)  # 从空字符串开始回溯，初始左右括号数量都为0
        return res

    # todo 组合回溯 93. 复原 IP 地址
    def restoreIpAddresses(s):
        res = []  # 存储所有可能的有效IP地址
        path = []

        def backtrack(start):
            if start == len(s) and len(path) == 4:  # 当字符串遍历完且路径中有4个整数时，表示生成了一个有效IP地址
                res.append(".".join(path))
                return

            if start == len(s) or len(path) == 4:  # 如果路径中已有4个整数或者字符串已遍历完，则回溯
                return

            for i in range(start, min(start + 3, len(s))):  # 每个整数最多有3位数
                if i > start and s[start] == '0':  # 如果当前整数以0开头，则只能是0本身
                    break

                num = int(s[start:i + 1])  # 当前整数的值
                if num <= 255:  # 当前整数的值在0到255之间时，可以加入路径
                    path.append(str(num))
                    backtrack(i + 1)  # 递归进入下一层
                    path.pop()  # 回溯，将当前整数从路径中移除

        backtrack(0)  # 从索引0开始回溯，初始路径为空列表
        return res

    # TODO 全排列回溯：
    def permute(self, nums):
        res = []  # 存储所有全排列的结果

        def backtrack(path, used):
            if len(path) == len(nums):  # 当路径长度等于数组长度时，表示已经生成了一个全排列
                res.append(path[:])
                return

            for i in range(len(nums)):
                if not used[i]:  # 如果数字未被使用过，则可以加入当前路径
                    path.append(nums[i])
                    used[i] = True
                    backtrack(path, used)  # 递归进入下一层
                    path.pop()  # 回溯，将当前数字从路径中移除
                    used[i] = False

        used = [False] * len(nums)  # 记录数字是否被使用过的列表
        backtrack([], used)  # 从空路径开始回溯
        return res


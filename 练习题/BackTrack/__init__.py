#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/5 14:17
# @Author  : MinisterYU
# @File    : __init__.py.py


# 给一个数组，和一个目标值，求出数组中所有元素之和等于目标值的组合结果
# 这个是所有元素组合， 所以dfs的时候，下标直接传递，退出条件为元素组合 = 目标
def find_sum_rs(nums=[1, 2, 3, 6], target=9):
    nums_l = len(nums)
    ans = []
    path = []

    def dfs(i, target_):
        # 退出条件，当sum值等于target的时候，就把结果返回到集合中，
        if target_ == 0:
            ans.append(path[:])
            return

        for j in range(i, nums_l):
            # 如果元素值大于target,则不取
            if nums[j] > target_:
                continue
            # 选择元素加入路径
            path.append(nums[j])
            # 因为元素可以重复选择，所以直接按当前的元素填入再搜索
            dfs(j, target_ - nums[j])
            path.pop()

    dfs(0, target)
    return ans


# print(find_sum_rs(nums=[2, 3, 6], target=9))


# 找到一个字符串中的所有回文字符串
# 进入下个节点是上一个节点截取后的队列
# abacbabc [a.... aba... cbabc..]

def find_palindrome(strs):
    strs_l = len(strs)
    ans = []
    path = []

    def dfs(i):
        # 退出条件
        if i == strs_l:
            ans.append(path[:])
            return

        for j in range(i, strs_l):
            # 判断是否是回文:
            if strs[i:j + 1] == strs[i:j + 1][::-1]:
                path.append(strs[i:j + 1])
                dfs(j + 1)
                path.pop()

    dfs(0)
    return ans


# print(find_palindrome('aba'))



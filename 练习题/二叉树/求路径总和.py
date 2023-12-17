#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 19:02
# @Author  : MinisterYU
# @File    : 求路径总和.py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
'''
TODO 求路径总和：给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
https://leetcode.cn/problems/path-sum-iii/description/
'''
class Solution:
    def pathSum(self, root, target):
        # 前缀和为0的一条路径
        prefixSumCount = {0: 1}

        res = self.recursionPathSum(root, prefixSumCount, target, 0)

        return res

    def recursionPathSum(self, node, prefixSumCount, target, currSum):
        # 1.递归终止条件
        if not node:
            return 0
        # 2.本层要做的事情
        res = 0
        # 当前路径上的和
        currSum += node.val

        # ---核心代码
        # 看看root到当前节点这条路上是否存在节点前缀和加target为currSum的路径
        # 当前节点->root节点反推，有且仅有一条路径，如果此前有和为currSum-target
        # 而当前的和又为currSum,两者的差就肯定为target了
        # currSum-target相当于找路径的起点，起点的sum+target=currSum，当前点到起点的距离就是target
        res += prefixSumCount.get(currSum - target, 0)
        print(currSum - target)
        # 更新路径上当前节点前缀和的个数
        prefixSumCount[currSum] = prefixSumCount.get(currSum, 0) + 1

        # 进入下一层
        res += self.recursionPathSum(node.left, prefixSumCount, target, currSum)
        res += self.recursionPathSum(node.right, prefixSumCount, target, currSum)

        # 回到本层，恢复状态，去除当前节点的前缀和数量
        prefixSumCount[currSum] -= 1
        return res

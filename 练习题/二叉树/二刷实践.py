#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 10:41
# @Author  : MinisterYU
# @File    : 二刷实践.py
import collections

from 练习题.二叉树 import TreeNode, printTree, arrayToTree
from 练习题.链表 import ListNode
from typing import Optional, List
from collections import deque


class Solution:

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        # 95. 不同的二叉搜索树 II
        # https://leetcode.cn/problems/unique-binary-search-trees-ii/description/
        if n == 0:
            return None

        def traverse(start, end):
            # 如果前面的节点大于后面的节点值了，则返回
            if start > end:
                return [None]
            res = []
            # 遍历
            for i in range(start, end + 1):
                # 左边和右边分别遍历
                left = traverse(start, i - 1)
                right = traverse(i + 1, end + 1)
                for left_node in left:
                    # 构建二叉树
                    for right_node in right:
                        root = TreeNode(val=i)
                        root.left = left_node
                        root.right = right_node
                        res.append(root)
            return res

        return traverse(1, n)

    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # 1123. 最深叶节点的最近公共祖先
        # https://leetcode.cn/problems/lowest-common-ancestor-of-deepest-leaves/
        self.ans = None
        self.max_depth = -1

        def traverse(root, depth):
            if not root:
                self.max_depth = max(self.max_depth, depth)  # 维护全局最大深度
                return depth
            # 计算左右叶子深度
            left_depth = traverse(root.left, depth + 1)
            right_depth = traverse(root.right, depth + 1)

            if left_depth == right_depth == self.max_depth:
                self.ans = root

            return max(left_depth, right_depth)
        traverse(root, 0)
        return self.ans


if __name__ == '__main__':
    pass

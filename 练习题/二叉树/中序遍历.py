#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/27 22:23
# @Author  : MinisterYU
# @File    : 中序遍历.py
from 练习题.二叉树 import TreeNode
from typing import Optional, List
from collections import deque


class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 递归解法
        self.res = []
        self.traverse(root)
        return self.res

    def traverse(self, root):
        if not root:
            return
        self.traverse(root.left)
        self.res.append(root.val)
        self.traverse(root.right)

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 遍历解法
        if not root:
            return []
        res = []
        left = self.inorderTraversal(root.left)
        res.append(root.val)
        right = self.inorderTraversal(root.right)
        return left + res + right

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 递推解法
        if not root:
            return []

        stack = []
        res = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            node = stack.pop()
            res.append(node.val)

            root = node.right
        return res

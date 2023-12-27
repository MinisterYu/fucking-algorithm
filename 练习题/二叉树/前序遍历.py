#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/27 21:46
# @Author  : MinisterYU
# @File    : 前序遍历.py
from 练习题.二叉树 import TreeNode
from typing import Optional, List
from collections import deque


class Solution:
    def preorderTraversal_递推解法(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        stack = deque()
        res = []
        stack.append(root)
        while stack:
            node = stack.pop()
            # 先处理根节点
            res.append(node.val)
            # 压栈，后进先出，所以先进右后进左，保障出来的是先左后右
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

    def preorderTraversal_遍历解法(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res
        # 前序遍历的结果，root.val 在第一个
        res.append(root.val)
        # 利用函数定义，后面接着左子树的前序遍历结果
        res.extend(self.preorderTraverse(root.left))
        # 利用函数定义，最后接着右子树的前序遍历结果
        res.extend(self.preorderTraverse(root.right))

        return res

    def preorderTraversal_分解解法(self, root: Optional[TreeNode]) -> List[int]:
        res = []

        def traverse(root):
            if not root:
                return
            # 前序位置
            res.append(root.val)
            traverse(root.left)
            traverse(root.right)

        traverse(root)
        return res

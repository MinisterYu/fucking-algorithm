#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 15:22
# @Author  : MinisterYU
# @File    : __init__.py.py

from typing import Optional, List
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Solution:

    # TODO 2叉树 前续遍历
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 前序 中 左 右
        if not root:
            return []
        res, stack = deque(), deque()
        stack.append(root)
        while stack:
            # 右边出栈
            node = stack.pop()
            res.append(node.val)

            if node.right:
                stack.append(node.right)
            # 右边入栈
            if node.left:
                stack.append(node.left)

        return res

    # TODO 2叉树 后续遍历
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 后序： 左 右 中
        if not root:
            return []
        res, stack = deque(), deque()
        stack.append(root)
        while stack:
            # 右边出栈
            node = stack.pop()
            res.appendleft(node.val)
            # 右边入栈
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return res

    # TODO 2叉树 中续遍历
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 中序 ： 左、中、右
        if not root:
            return []
        res, stack = deque(), deque()
        while stack or root:
            # 右边入栈，左节点入栈
            while root:
                stack.append(root)
                root = root.left

            # 右边出栈
            node = stack.pop()
            res.append(node.val)
            root = node.right
        return res

    #TODO  二叉树层序遍历
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

        if not root:
            return []

        result = []
        queue = deque(root)

        while queue:
            level = []
            level_size = len(queue)

            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result += level

        return result

    # TODO N叉树 前续遍历
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        res, stack = deque(), deque()
        stack.append(root)
        while stack:
            # 左边出栈
            node = stack.pop()
            res.append(node.val)

            for child in reversed(node.children):
                stack.append(child)
        return res

    # TODO N叉树 后续遍历
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        res, stack = deque(), deque()
        stack.append(root)
        while stack:
            node = stack.pop()
            res.appendleft(node.val)

            for child in node.children:
                stack.append(child)
        return res


if __name__ == '__main__':
    solution = Solution()
    solution.levelOrder()
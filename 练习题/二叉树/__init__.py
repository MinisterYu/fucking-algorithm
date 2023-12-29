#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 19:02
# @Author  : MinisterYU
# @File    : __init__.py.py

from collections import deque
import heapq


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# def buildTree(nums):
#     if not nums:
#         return None
#     return buildTreeHelper(nums, 0, len(nums) - 1)

def __printTreeHelper(node, prefix, is_left):
    if node:
        print(prefix, end="")
        print("├── " if is_left else "└── ", end="")
        print(node.val)

        if node.left or node.right:
            __printTreeHelper(node.left, prefix + ("│   " if is_left else "    "), True)
            __printTreeHelper(node.right, prefix + ("│   " if is_left else "    "), False)


def printTree(root):
    if not root:
        return
    __printTreeHelper(root, "", True)


def arrayToTree(nums):
    if not nums:
        return None

    root = TreeNode(nums[0])
    queue = deque([root])

    i = 1
    while queue and i < len(nums):
        node = queue.popleft()

        if nums[i] is not None:
            node.left = TreeNode(nums[i])
            queue.append(node.left)
        i += 1

        if i < len(nums) and nums[i] is not None:
            node.right = TreeNode(nums[i])
            queue.append(node.right)
        i += 1

    return root


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return 'null'

        return f'{root.val},{self.serialize(root.left)},{self.serialize(root.right)}'

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        def build(nodes):
            if nodes[0] == 'null':
                nodes.pop(0)
                return None

            root = TreeNode(nodes.pop(0))
            root.left = build(nodes)
            root.right = build(nodes)

            return root

        return build(data.split(','))

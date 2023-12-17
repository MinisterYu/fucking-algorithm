#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 19:02
# @Author  : MinisterYU
# @File    : __init__.py.py
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def buildTree(nums):
    if not nums:
        return None
    return buildTreeHelper(nums, 0, len(nums) - 1)


def buildTreeHelper(nums, start, end):
    if start > end:
        return None

    mid = (start + end) // 2
    root = TreeNode(nums[mid])

    root.left = buildTreeHelper(nums, start, mid - 1)
    root.right = buildTreeHelper(nums, mid + 1, end)

    return root


def treeToArray(root):
    result = []
    treeToArrayHelper(root, result)
    return result


def treeToArrayHelper(node, result):
    if not node:
        return

    result.append(node.val)
    treeToArrayHelper(node.left, result)
    treeToArrayHelper(node.right, result)

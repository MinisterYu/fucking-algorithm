#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 11:25
# @Author  : MinisterYU
# @File    : 二叉树的问题.py
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree_中序和后序遍历构建二叉树(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not postorder or not inorder:
            return None

        root_val = postorder.pop()
        root = TreeNode(val=root_val)

        root_index = inorder.index(root_val)
        # 先右后左  --> 左右中  --> 中右左
        root.right = self.buildTree(inorder[root_index + 1:], postorder)
        root.left = self.buildTree(inorder[:root_index], postorder)

        return root

    def buildTree__中序和前序遍历构建二叉树(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None

        root_val = preorder.pop(0)
        root = TreeNode(val=root_val)
        root_index = inorder.index(root_val)
        # 先左后右 --> 中左右
        root.left = self.buildTree(preorder, inorder[:root_index])
        root.right = self.buildTree(preorder, inorder[root_index + 1:])

        return root

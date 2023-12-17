#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 19:52
# @Author  : MinisterYU
# @File    : 简单二叉树遍历.py
from 练习题.二叉树 import TreeNode, buildTree, treeToArray
from typing import Optional


class Solution:

    # 统计二叉树的节点个数
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        left = self.countNodes(root.left)
        right = self.countNodes(root.right)

        return left + right + 1

    # 判断相同的树
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p or not q:
            return p is q

        if p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right):
            return True

        return False

    # 二叉树的最大深度
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        max_left = self.maxDepth(root.left)
        max_right = self.maxDepth(root.right)

        return max(max_left, max_right) + 1

    # 二叉树的最小深度
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        # 如果左子树为空，递归计算右子树的最小深度
        if not root.left:
            return self.minDepth(root.right) + 1

        # 如果右子树为空，递归计算左子树的最小深度
        if not root.right:
            return self.minDepth(root.left) + 1

        # 如果左右子树都不为空，递归计算左右子树的最小深度，取较小值
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

    # 翻转二叉树
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        root.left, root.right = root.right, root.left

        self.invertTree(root.left)
        self.invertTree(root.right)

        return root

    def is_leef(self, root):
        if root and not root.left and not root.right:
            return True
        return False

    # 左叶子之和
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        if root.left and self.is_leef(root.left):
            return root.left.val + self.sumOfLeftLeaves(root.right)

        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    def diameterOfBinaryTree(self, root):
        self.max_diameter = 0
        self.depth(root)
        return self.max_diameter

    def depth(self, node: TreeNode):
        if not node:
            return 0

        left_depth = self.depth(node.left)
        right_depth = self.depth(node.right)

        self.max_diameter = max(self.max_diameter, left_depth + right_depth)

        return max(left_depth, right_depth) + 1


if __name__ == '__main__':
    solution = Solution()
    # res1 = solution.countNodes(buildTree([1, 2, 3, 4, 5, 6]))
    # res2 = solution.isSameTree(buildTree([1, 2, 3, 5, 4, 6]), buildTree([1, 2, 3, 4, 5, 6]))
    # res3 = solution.maxDepth(buildTree([1, 2, 3, 4, 5, 6]))

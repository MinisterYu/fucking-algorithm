#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 19:52
# @Author  : MinisterYU
# @File    : 简单二叉树遍历.py
from 练习题.二叉树 import TreeNode, arrayToTree
from typing import Optional, List

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


    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # TODO 二叉树的最大深度
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

    # 左叶子之和
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        def is_leef(root):
            if root and not root.left and not root.right:
                return True
            return False

        if root.left and is_leef(root.left):
            return root.left.val + self.sumOfLeftLeaves(root.right)

        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    def diameterOfBinaryTree(self, root):
        self.max_diameter = 0

        def depth(node: TreeNode):
            if not node:
                return 0

            left_depth = depth(node.left)
            right_depth = depth(node.right)

            self.max_diameter = max(self.max_diameter, left_depth + right_depth)

            return max(left_depth, right_depth) + 1

        depth(root)
        return self.max_diameter

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1:
            return root2

        if not root2:
            return root1

        val = root1.val + root2.val
        left = self.mergeTrees(root1.left, root2.left)
        right = self.mergeTrees(root1.right, root2.right)
        return TreeNode(val, left=left, right=right)

    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        if not root:
            return []
        res = []

        def dfs(root, deptth):
            if not root:
                return None

            if len(res) == deptth:
                res.append([])

            dfs(root.left, deptth + 1)
            dfs(root.right, deptth + 1)

            return res[deptth].append(root.val)

        dfs(root, 0)
        return [sum(i) / len(i) for i in res]

    # 找到最底层 最左边 节点的值

    def findBottomLeftValue(self, root):

        def dfs(node, depth):
            if not node:
                return None, -1

            # 如果当前节点为叶子节点，返回节点的值和深度
            if not node.left and not node.right:
                return node.val, depth

            # 递归处理左子节点和右子节点
            left_value, left_depth = dfs(node.left, depth + 1)
            right_value, right_depth = dfs(node.right, depth + 1)

            # 比较左子节点和右子节点的深度，返回深度较大的节点值和深度
            if left_depth >= right_depth:
                print(
                    f'left_depth {left_depth} left_value {left_value} | right_depth {right_depth} right_value {right_value}')
                return left_value, left_depth
            else:
                print(
                    f'left_depth {left_depth} left_value {left_value} | right_depth {right_depth} right_value {right_value}')
                return right_value, right_depth

        max_value = dfs(root, 0)[0]
        return max_value


if __name__ == '__main__':
    solution = Solution()
    # res1 = solution.countNodes(buildTree([1, 2, 3, 4, 5, 6]))
    # res2 = solution.isSameTree(buildTree([1, 2, 3, 5, 4, 6]), buildTree([1, 2, 3, 4, 5, 6]))
    # res3 = solution.maxDepth(buildTree([1, 2, 3, 4, 5, 6]))
    # res4 = solution.mergeTrees(buildTree([1, 2, 3]), buildTree([2, 3]))
    res5 = solution.findBottomLeftValue(arrayToTree([1, 2, 3, 4, None, 5, 6, None, None, 7]))
    print(res5)

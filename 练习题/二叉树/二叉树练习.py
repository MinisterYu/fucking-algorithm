#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/29 14:32
# @Author  : ministeryu
# @File    : 二叉树练习.py

import collections
import math
from typing import List, Optional
from collections import defaultdict, deque, Counter


class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class Solution:

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        '''二叉树中序遍历'''
        if not root:
            return []
        stack = []
        ans = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            node = stack.pop()
            ans.append(node.val)
            root = node.right
        return ans

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        '''给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。'''
        if n == 0:
            return None

        def dfs(start, end):
            if start > end:
                return [None]
            res = []
            for i in range(start, end + 1):
                left = dfs(start, i - 1)
                right = dfs(i + 1, end)
                for left_node in left:
                    for right_node in right:
                        root = TreeNode(i)
                        root.left = left_node
                        root.right = right_node
                        res.append(root)
            return res

        return dfs(1, n)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def dfs(root, left, right):
            if not root:
                return True
            if root.val <= left or root.val >= right:
                return False
            left_val = dfs(root.left, left, root.val)
            right_val = dfs(root.right, root.val, right)
            return left_val and right_val

        return dfs(root, -inf, inf)

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res = []
        stack = deque([root])
        order = False
        while stack:
            n = len(stack)
            temp = deque()
            for _ in range(n):
                node = stack.popleft()
                if order:
                    temp.append(node.val)
                else:
                    temp.appendleft(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
            res.append(list(temp))

        return list(res)

    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res = []
        stack = deque([root])

        while stack:
            n = len(stack)
            temp = deque()
            for _ in range(n):
                node = stack.popleft()
                temp.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
            res.append(list(temp))

        return list(res)[::-1]

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        '''
        给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
        叶子节点 是指没有子节点的节点。
        '''
        self.res = []

        def dfs(root, currSum, curPtah):
            if not root:
                return []
            currSum += root.val
            curPtah.append(root.val)
            if currSum == targetSum and not root.left and not root.right:
                self.res.append(curPtah[:])
            dfs(root.left, currSum, curPtah)
            dfs(root.right, currSum, curPtah)
            currSum -= root.val
            curPtah.pop()

        dfs(root, 0, [])
        return self.res

    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        stack = deque([root])
        while stack:
            n = len(stack)
            for i in range(n):
                node = stack.popleft()
                if i != n - 1:
                    node.next = stack[0]
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        return root

    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        self.res = 0

        def traverse(root, curnum):
            if not root:
                return
            curnum = curnum * 10 + root.val
            if not root.left and not root.right:
                self.res += curnum
            traverse(root.left, curnum)
            traverse(root.right, curnum)
            curnum = (curnum - root.val) // 10

        traverse(root, 0)
        return self.res

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if p == root or q == root:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left or right

    def pathSumIII(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.res = 0
        self.prefix = {0: 1}

        def dfs(root, currSum):
            if not root:
                return
            currSum += root.val
            if currSum - targetSum in self.prefix:
                self.res += self.prefix[currSum - targetSum]
            dfs(root.left, currSum)
            dfs(root.right, currSum)
            currSum -= root.val

        dfs(root, 0)
        return self.res

    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        self.res = []
        stack = deque([root])
        while stack:
            n = len(stack)
            max_res = -inf
            for i in range(n):
                node = stack.popleft()
                max_res = max(max_res, node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
            self.res.append(max_res)
        return self.res

    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        self.memo = collections.Counter()
        self.result = []

        def traverse(root):
            if not root:
                return '#'
            left = traverse(root.left)
            right = traverse(root.right)
            sub_tree = f"{root.val},{left},{right}"
            self.memo[sub_tree] += 1
            if self.memo[sub_tree] >= 2:
                self.result.append(root)
            return sub_tree

        traverse(root)
        return self.result

    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        self.ans = 0

        # 统计node下的coins总数，和node总数
        def dfs(root):
            if not root:
                return 0, 0
            nodes_left, coins_left = dfs(root.left)
            nodes_right, coins_right = dfs(root.right)
            coins = coins_left + coins_right + root.val
            nodes = nodes_left + nodes_right + 1
            self.ans += abs(coins - nodes)
            return nodes, coins

        dfs(root)
        return self.ans

    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        self.cnt = {}

        def dfs(root, cursum, depth, path):
            if not root:
                return
            cursum += root.val * (10 ** depth )
            path.append(root.val)
            if not root.left and not root.right:
                self.cnt[cursum] = path
            dfs(root.left, cursum, depth + 1, path)
            dfs(root.right, cursum, depth + 1, path)
            path.pop()
            cursum -= root.val * depth

        dfs(root, 0, 1, [])
        min_path = self.cnt[min(self.cnt.keys())]
        return ''.join(chr(ord('a') + i) for i in min_path[::-1])


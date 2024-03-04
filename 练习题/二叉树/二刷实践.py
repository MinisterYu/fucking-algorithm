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
import math
import heapq

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

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        queue = collections.deque([s])
        visited = set()
        while queue:
            rest = queue.popleft()
            if not rest:
                return True

            for word in wordDict:
                if rest.startswith(word):
                    # visited.add(word)
                    queue.append(rest[len(word):])
        print(visited)
        return False

    def lastStoneWeightII(self, stones: List[int]) -> int:
        target = sum(stones) // 2
        dp = [0] * (target + 1)
        for i in range(len(stones)):
            for j in range(target, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])

        return sum(stones) - 2 * dp[-1]

    def nthUglyNumber(self, n: int) -> int:
        # p2, p3, p5 = 1, 1, 1
        # v2, v3, v5 = 1, 1, 1
        # ugly = [0] * (n + 1)
        # p = 1
        # while p <= n:
        #     min_v = min(v2, v3, v5)
        #     ugly[p] = min_v
        #     p += 1
        #     if min_v == v2:
        #         v2 = 2 * ugly[p2]
        #         p2 += 1
        #     if min_v == v3:
        #         v3 = 3 * ugly[p3]
        #         p3 += 1
        #     if min_v == v5:
        #         v5 = 5 * ugly[p5]
        #         p5 += 1
        # print(ugly)

        heap = [1]
        index = 0
        visited = set()
        while index < n:
            index += 1
            val = heapq.heappop(heap)
            for i in [2, 3, 5]:
                if i * val not in visited:
                    visited.add(i * val)
                    heapq.heappush(heap, i * val)

        return val
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        n = len(books)
        dp = [math.inf] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            height, width = 0, shelfWidth
            for j in range(i, -1, -1):
                width -= books[j][0]
                if width < 0:
                    break
                height = max(height, books[j][1])
                dp[i + 1] = min(dp[i + 1], dp[j] + height)
        return dp[-1]

    def isCompleteTree(self, root: TreeNode) -> bool:
        # 判断是否是完全二叉树
        # https://www.nowcoder.com/practice/8daa4dff9e36409abba2adbe413d6fae?tpId=295&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Foj
        if not root:
            return True

        queue = deque()
        queue.append(root)

        while queue:
            node = queue.popleft()

            if node:
                queue.append(node.left)
                queue.append(node.right)
            else: # 如果加入的是空节点
                # 有未访问到的非空节点，则树存在空洞，为非完全二叉树
                while queue: # 如果空节点后面还有非空节点，则为非完全二叉树
                    if queue.popleft():
                        return False

        return True

if __name__ == '__main__':
    so = Solution()
    # print(so.wordBreak("applepenapple", ["apple","pen"]))
    so.nthUglyNumber(10)

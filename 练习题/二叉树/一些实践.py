#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/27 23:22
# @Author  : MinisterYU
# @File    : 一些实践.py
import collections

from 练习题.二叉树 import TreeNode, printTree, arrayToTree
from 练习题.链表 import ListNode
from typing import Optional, List
from collections import deque


# root = arrayToTree([1, 2, 3, 4, None, None, 7])
# printTree(root)


class Solution:

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # https://leetcode.cn/problems/diameter-of-binary-tree/
        # 543. 二叉树的直径
        # 找到最左边和最右边的节点
        self.max_diameter = 0

        def traverse(root) -> int:
            if not root:
                return 0

            left = traverse(root.left)
            right = traverse(root.right)
            self.max_diameter = max(self.max_diameter, right + left)  # 比较左右深度之和，与当前最大深度之间哪个大

            return max(left, right) + 1  # 找到左右节点的深度

        traverse(root)
        print(self.max_diameter)
        return self.max_diameter

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        # https://leetcode.cn/problems/unique-binary-search-trees-ii/
        # 95. 不同的二叉搜索树 II
        def traverse(start, end) -> List[Optional[TreeNode]]:
            if start > end:
                return [None]
            result = []
            for i in range(start, end + 1):
                left_nodes = traverse(start, i - 1)
                right_nodes = traverse(i + 1, end)

                for left_node in left_nodes:
                    for right_node in right_nodes:
                        # 根节点要在这里创建，才能构建子树，如果在 for i in range(start, end + 1) 这里构建，那么所有子节点都会加在一个根节点下
                        root = TreeNode(i)
                        root.left = left_node
                        root.right = right_node
                        result.append(root)
            return result

        return traverse(1, n)

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/
        self.res = []

        def traverse(root, depth):
            if not root:
                return []
            if depth == len(self.res):
                self.res.append([])
            if depth % 2 == 0:
                self.res[depth].append(root.val)
            else:
                self.res[depth].insert(0, root.val)
            traverse(root.left, depth + 1)
            traverse(root.right, depth + 1)

        traverse(root, 0)
        return self.res

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # https://leetcode.cn/problems/maximum-depth-of-binary-tree/
        self.max_dep = 0

        def traverse(node, depth):
            if not node:
                return
            depth += 1
            if not node.right and not node.left:
                self.max_dep = max(self.max_dep, depth)
            traverse(node.left, depth)
            traverse(node.right, depth)
            depth -= 1

        traverse(root, 0)
        return self.max_dep

    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        # https://leetcode.cn/problems/binary-tree-level-order-traversal-ii/
        if not root:
            return []
        stack = deque([root])
        res = []
        while stack:
            tmp = deque()
            for _ in range(len(stack)):
                node = stack.popleft()
                tmp.append(node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
            res.insert(0, list(tmp))
        return res

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        # https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/
        if not nums:
            return None
        mid = len(nums) // 2
        left = self.sortedArrayToBST(nums[:mid])
        right = self.sortedArrayToBST(nums[mid + 1:])
        return TreeNode(val=nums[mid], left=left, right=right)

    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        # https://leetcode.cn/problems/convert-sorted-list-to-binary-search-tree/
        def mid_node(head: ListNode, last):
            fast = slow = head
            while fast != last and fast.next != last:
                fast = fast.next.next
                slow = slow.next

            return slow

        def build(head, last):
            if head == last:
                return None
            mid = mid_node(head, last)
            root = TreeNode(val=mid.val)
            root.left = build(head, mid)
            root.right = build(mid.next, last)
            return root

        return build(head, None)

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def traverse(root):
            if not root:
                return 0
            left = traverse(root.left)
            right = traverse(root.right)
            # 后序遍历，前面的已经计算过了
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            else:
                return max(left, right) + 1

        return traverse(root) >= 0

    pre_node = None

    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/
        """
        if not root:
            return None

            # 递归展开左子树和右子树
        left = self.flatten(root.left)
        right = self.flatten(root.right)

        # 将左子树插入到根节点和右子树之间
        if left:
            root.right = left
            root.left = None

            # 找到左子树的最右侧节点
            while left.right:
                left = left.right

            # 将右子树插入到左子树的最右侧节点之后
            left.right = right

        return root

        # 第二种解法，遍历

        if not root:
            return None

        stack = [root]
        prev = None
        while stack:
            curr = stack.pop()
            print(curr.val)
            if curr.right:
                stack.append(curr.right)
            if curr.left:
                stack.append(curr.left)
            if prev:
                prev.right = curr
                prev.left = None
            prev = curr
            # printTree(root)

    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        # https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/
        if not root:
            return None

        def traverse(node1, node2):
            if not node1 and not node2:
                return

            node1.next = node2
            traverse(node1.left, node1.right)

            traverse(node2.left, node2.right)
            traverse(node1.right, node2.left)

        traverse(root.left, root.right)
        return root

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        # https://leetcode.cn/problems/path-sum-iii/
        # 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
        # 1、 前缀和，把路径上的所有节点和都记录下来，放在hash表中
        # 2、 进入节点的时候，targetsum - node.val 如果出现在hash表中，则答案 + 1
        # 3、
        self.cur_sum = 0
        self.prefix_map = {0: 1}  # 前缀和为0 的一条路径
        self.res = 0

        def preSum(root, targetSum):
            if not root:
                return 0

            self.cur_sum += root.val  # 累计路径和
            # 记录结果，在hash表中看是否能找到 targetSum - self.cur_sum 为key 的路径值
            self.res += self.prefix_map.get(targetSum - self.cur_sum, 0)

            # hash表中记录当前路径的值
            self.prefix_map[self.cur_sum] = self.prefix_map.get(self.cur_sum, 0) + 1
            # 进入到当前节点的下面节点，递归去找
            preSum(root.left, targetSum)
            preSum(root.right, targetSum)
            # 弹出
            self.prefix_map[self.cur_sum] = self.prefix_map.get(self.cur_sum, 0) - 1

        preSum(root, targetSum)
        return self.res

    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        # https://leetcode.cn/problems/distribute-coins-in-binary-tree/
        # 移动硬币的次数 = 移出的总次数
        # 移出的次数分解子问题， 在一颗子树上，硬币总数 - 节点总数
        # 硬币总数 = 左边子硬币和 + 右边子硬币和 + 当前节点硬币数
        # 节点总数 = 左边子节点和 + 右边子节点和 + 1

        self.ans = 0  # 统计硬币移动总次数

        # 定义：函数统计当前node下，node和coins的总数
        def traverse(node) -> (int, int):
            if not node:
                return 0, 0
            # 分别统计左右边
            coins_left, nodes_left = traverse(node.left)
            coins_right, nodes_right = traverse(node.right)
            # 硬币总数
            conis = coins_left + coins_right + node.val
            # 节点总数
            nodes = nodes_left + nodes_right + 1

            # 移出或者移入本节点的次数记录
            self.ans += abs(conis - nodes)

            return conis, nodes

        traverse(root)
        return self.ans

    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        # https://leetcode.cn/problems/find-duplicate-subtrees/
        # 返回二叉树的所有子树
        self.memo = {}
        self.result = []

        def traverse(root):
            if not root:
                return 'null'
            left = traverse(root.left)
            right = traverse((root.right))
            sub_tree = f'{root.val},{left},{right}'
            self.memo[sub_tree] = self.memo.get(sub_tree, 0) + 1
            if self.memo[sub_tree] == 2:
                self.result.append(root)

            return sub_tree

        traverse(root)
        print(self.memo)
        return self.result

    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        # https://leetcode.cn/problems/most-frequent-subtree-sum/submissions/492008850/
        if not root:
            return []
        self.memo = collections.Counter()

        def traverse(root):
            if not root:
                return 0

            left = traverse(root.left)
            right = traverse(root.right)
            self.memo[root.val + left + right] += 1
            return root.val + left + right

        traverse(root)
        max_count = max(self.memo.values())
        print(self.memo)
        return [i for i in self.memo if self.memo[i] == max_count]

    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        # https://leetcode.cn/problems/smallest-string-starting-from-leaf/
        '''
        给定一颗根结点为 root 的二叉树，树中的每一个结点都有一个 [0, 25] 范围内的值，分别代表字母 'a' 到 'z'。
        返回 按字典序最小 的字符串，该字符串从这棵树的一个叶结点开始，到根结点结束。
        '''
        if not root:
            return ""

        def dfs(node, path):
            if not node:
                return ""

            # 将当前节点的值转换为对应的字母，并添加到路径中
            path = chr(node.val + ord('a')) + path

            if not node.left and not node.right:
                # 如果当前节点是叶节点，则返回当前路径作为一个候选结果
                return path

            # 递归地遍历左子树和右子树，并返回字典序最小的路径
            left_path = dfs(node.left, path)
            right_path = dfs(node.right, path)

            if left_path and right_path:
                # 如果左子树和右子树都有路径，则返回字典序最小的路径
                return min(left_path, right_path)
            elif left_path:
                # 如果只有左子树有路径，则返回左子树的路径
                return left_path
            else:
                # 如果只有右子树有路径，则返回右子树的路径
                return right_path

        return dfs(root, "")

    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        res = level = 1
        max_count = root.val
        stack = [root]
        while stack:
            n = len(stack)
            cur_count = 0
            for _ in range(n):
                node = stack.pop()
                cur_count += node.val
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
            if max_count < cur_count:
                max_count = cur_count
                res = level
                print(max_count, res)
            level += 1

        print(res)

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        def traverse(start, end):
            if start > end:
                return [None]
            res = []
            for i in range(start, end):
                left_nodes = traverse(start, i - 1)
                right_nodes = traverse(i + 1, end)

                for left in left_nodes:
                    for right in right_nodes:
                        root = TreeNode(val=i, left=left, right=right)
                        res.append(root)
            return res

        return traverse(1, n + 1)

    def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
        self.res = -1

        def traverse(root, cur):
            if not root:
                return
            if root.val != cur:
                if self.res == -1:
                    self.res = root.val
                else:
                    self.res = min(root.val, self.ans)
                    return
            traverse(root.left, cur)
            traverse(root.right, cur)

        traverse(root, root.val)
        return self.res

    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        n = len(arr)
        res = [0] * n
        for i in range(k):
            res[i] = max(arr[:k])

        for i in range(k, n - k):
            res[i] = max(arr[i: i + k])

        for i in range(n - k, n):
            res[i] = max(arr[n - k: n])

        print(res)

    def isPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        while left < right:
            if not s[left].isalnum():
                left += 1
            if not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True



if __name__ == '__main__':
    so = Solution()
    # tree = arrayToTree([3, 9, 20, None, None, 15, 7])
    # tree = arrayToTree([1, 2, 3, 4, None, None, 5])
    # printTree(tree)
    # res = so.zigzagLevelOrder(tree)
    # res = so.maxDepth(tree)
    # res = so.levelOrderBottom(tree)
    # print(res)
    # root = arrayToTree([1, 2, 2, 3, 3, None, None, 4, 4])
    # so.isBalanced(root)

    # root = arrayToTree([1, 2, 5, 3, 4, None, 6])
    # root = arrayToTree([1, 2, 5, 3, 4, 6, 7])
    # printTree(root)
    # # so.flatten(root)
    # demo(root)
    # root = arrayToTree([1, 2, 3, 4, 5, None, 6, None, None, None, None, None, 7])
    # printTree(root)
    # q = deque([root])
    # while q:
    #     node = q.popleft()
    #     if node.right:
    #         q.append(node.right)
    #     if node.left:
    #         q.append(node.left)
    # print(node.val)
    # root = arrayToTree([2,1,11,11,None,1])
    # printTree(root)
    # res = so.findDuplicateSubtrees(root)
    # print(res)
    # root = arrayToTree([5, 14, None, 1])
    # printTree(root)
    # so.findFrequentTreeSum(root)
    # root = so.sortedArrayToBST([1, 2, 3, 4, 5, 6, 7])
    # root = arrayToTree([4, 2, 1, 3, 6, 5, 7])
    # printTree(root)
    # so.smallestFromLeaf(arrayToTree([0, 1, 2, 3, 4, 3, 4]))
    # so.maxLevelSum(arrayToTree([989, None, 10250, 98693, -89388, None, None, None, -32127]))
    # s = so.isPalindrome("A man, a plan, a canal: Panama")
    # print(s)
    # so.findWords(board=[["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],
    #              words=["oath","pea","oaane","rain"])
    s1 = {'a', 'b'}
    s2 = {'b', 'c'}
    print(s1 & s2)
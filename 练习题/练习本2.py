#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/1 下午7:39
# @Author  : MinisterYU
# @File    : 练习本2.py
from multiprocessing import heap
from typing import List, Optional
from collections import defaultdict, deque, Counter
from math import inf
import heapq


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def preorder(self, root: Node) -> List[int]:
        '''
        N叉树前序遍历
        '''
        if not root:
            return []
        stack = deque()
        res = deque()
        stack.append(root)

        # 中\左\右
        while stack:
            node = stack.pop()
            res.append(node.val)
            for child in reversed(node.children):
                stack.append(child)
        return res

    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        给你一个链表的头节点 head 。移除每个右侧有一个更大数值的节点。返回修改后链表的头节点 head 。
        输入：head = [5,2,13,3,8]
        输出：[13,8]
        '''

        def reverseNodes(head):
            if not head or not head.next:
                return head

            next_ = reverseNodes(head.next)
            head.next.next = head
            head.next = None
            return next_

        head = reverseNodes(head)
        # [8,3,13,2,5]
        cur = head
        while cur and cur.next:
            if cur.val > cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return reverseNodes(head)

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        '''
        请根据每日 气温 列表 temperatures ，重新生成一个列表，要求其对应位置的输出为：
        要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。
        '''
        res = [0] * len(temperatures)
        stack = deque()
        for i in range(len(temperatures)):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                index = stack.pop()
                res[index] = i - index
            stack.append(i)
        return res

    def removeStars(self, s: str) -> str:
        stack = deque()
        for i in range(len(s)):
            if s[i] == '*':
                stack.pop()
            else:
                stack.append(s[i])
        return ''.join(stack)

    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 二叉树的前序遍历, 中、左、右
        if not root:
            return []
        stack = deque()
        res = deque()
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return list(res)

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        # 下一个更大的元素II
        '''给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。
        数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。
        如果不存在，则输出 -1 。
        '''
        stack = []
        res = [-1] * len(nums)
        for i in range(len(nums) * 2):
            while stack and nums[stack[-1] % len(nums)] < nums[i % len(nums)]:
                index = stack.pop()
                res[index % len(nums)] = nums[i % len(nums)]
            stack.append(i)

        return res

    def reorderList(self, head: Optional[ListNode]) -> None:
        ''' 重排链表
        输入：head = [1,2,3,4]
        输出：[1,4,2,3]'''

        if not head or not head.next:
            return head
        slow, fast = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        # 找到链表中间位置
        mid_head = slow.next
        slow.next = None

        # 倒排后面的链表
        pre = None
        cur_head = mid_head
        while cur_head:
            next_ = cur_head.next
            cur_head.next = pre
            pre = cur_head
            cur_head = next_

        sec_node = pre
        first_head = head
        sec_head = sec_node

        # 交叉
        while sec_head:
            first_head_next = first_head.next
            sec_head_next = sec_head.next

            first_head.next = sec_head
            sec_head.next = first_head_next

            first_head = first_head_next
            sec_head = sec_head_next

        return head

    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        # 链表中下一个更大的节点
        '''
        输入：head = [2,1,5]
        输出：[5,5,0]

        输入：head = [2,7,4,3,5]
        输出：[7,0,5,5,0]
        '''
        pre = None
        cur = head
        while cur:
            next_ = cur.next
            cur.next = pre
            pre = cur
            cur = next_
        cur_head = pre  # [5,3,4,7,2]

        stack = []
        ans = []
        while cur_head:
            while stack and stack[-1] <= cur_head.val:
                stack.pop()
            ans.append(stack[-1] if stack else 0)
            stack.append(cur_head.val)
            cur_head = cur_head.next
        return ans[::-1]

    def numSubmat(self, mat: List[List[int]]) -> int:
        # 统计全为1的矩形
        m, n = len(mat), len(mat[0])
        dp = [[0] * n for _ in range(m)]
        count = 0
        for i in range(m):
            for j in range(n):
                if mat[i][j] != 1:
                    continue
                dp[i][j] = dp[i][j - 1] + 1 if j > 0 else 1
                min_width = dp[i][j]
                for k in range(i, -1, -1):
                    min_width = min(min_width, dp[k][j])
                    count += min_width
        return count

    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 链表翻倍
        def reverse_head(head):
            if not head or not head.next:
                return head
            next_ = reverse_head(head.next)
            head.next.next = head
            head.next = None
            return next_

        cur_head = reverse_head(head)
        dummy_ret = ListNode(0)
        dummy = dummy_ret
        carry = 0
        while cur_head:
            val = (cur_head.val + carry) * 2
            res = val % 10
            carry = val // 10
            dummy.next = ListNode(val=res)
            cur_head = cur_head.next

        if carry:
            dummy.next = ListNode(val=carry)
        return reverse_head(dummy_ret)

    def smallestSubsequence(self, s: str) -> str:
        '''返回 s 字典序最小的子序列，该子序列包含 s 的所有不同字符，且只包含一次。'''
        stack = []
        counter = Counter(s)
        vist = set()
        for char in s:
            counter[char] -= 1
            if char in vist:
                continue
            while stack and stack[-1] > char and counter[stack[-1]]:
                stack.pop()
                vist.remove(stack[-1])

            vist.add(char)
            stack.append(char)
        return ''.join(stack)

    def makeGood(self, s: str) -> str:
        stack = []
        for char in s:
            if stack and stack[-1] != char and stack[-1].lower() == char.lower():
                stack.pop()
            else:
                stack.append(char)

        return ''.join(stack)

    def checkValidString(self, s: str) -> bool:
        stack = []
        s_stack = []
        for char in s:
            if char == '(':
                stack.append(char)
            elif char == '*':
                s_stack.append(char)
            else:
                if stack:
                    stack.pop()
                elif s_stack:
                    s_stack.pop()
                else:
                    return False
        print(stack)
        print(s_stack)
        while stack and s_stack:
            if stack[-1] > s_stack[-1]:
                return False
            stack.pop()
            s_stack.pop()
        return len(stack) == 0

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        stack = deque()
        res = []
        # 左中右
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            node = stack.popleft()
            res.append(node)
            if node.right:
                root = node.right
        return res

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # 验证搜索二叉树
        def dfs(root, left, right):
            if not root:
                return True
            if root.val <= left or root.val >= right:
                return False

            l = dfs(root.left, left, root.val)
            r = dfs(root.right, root.val, right)
            return l and r

        return dfs(root, -inf, inf)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # 验证对称二叉树
        if not root:
            return True

        def dfs(node1, node2):
            if not node1 and not node2:
                return True
            if not node1 or not node2:
                return False
            if node1.val != node2.val:
                return False

            return dfs(node1.left, node2.right) and dfs(node1.right, node2.left)

        return dfs(root.left, root.right)

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # 二叉树的层序遍历
        if not root:
            return []
        stack = deque([root])
        res = []
        while stack:
            n = len(stack)
            temp = []
            for _ in range(n):
                node = stack.pop()
                temp.append(node.val)
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)

            res.append(temp)
        return res

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 二叉树的最大深度

        def dfs(node, dpth):
            if not node:
                return dpth
            left = dfs(node.left, dpth + 1)
            right = dfs(node.right, dpth + 1)
            return max(left, right)

        return dfs(root, 0)

    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        # 2
        if i < 0:
            nums.reverse()
            return

        j = len(nums) - 1
        while j > i and nums[i] >= nums[j]:
            j -= 1
        # 3
        nums[i], nums[j] = nums[j], nums[i]
        # [1,3,2]
        nums[i + 1:] = reversed(nums[i + 1:])
        print(nums)

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        #
        ans = []
        path = []

        def dfs(index):
            if sum(path) > target:
                return
            if sum(path) == target:
                ans.append(path[:])
                return
            for i in range(index, len(candidates)):
                path.append(candidates[i])
                dfs(i)
                path.pop()

        dfs(0)
        return ans

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort(reverse=True)
        ans = []
        path = []

        def dfs(index):
            if sum(path) > target:
                return
            if sum(path) == target:
                ans.append(path[:])
                return

            for i in range(index, len(candidates)):
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                dfs(i + 1)
                path.pop()

        dfs(0)
        return ans

    def jump(self, nums: List[int]) -> int:
        # 返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
        dp = [0] * len(nums)
        dp[0] = 1
        jump = 0
        for i in range(1, len(nums)):
            while jump + nums[jump] < nums[i]:
                jump += 1
            dp[i] = dp[jump] + 1
        return dp[len(nums) - 1]

    def permute(self, nums: List[int]) -> List[List[int]]:
        ans = []
        path = []
        visited = [False] * len(nums)

        def dfs():
            if len(path) == len(nums):
                ans.append(path[:])
                return
            for i in range(len(nums)):
                if visited[i]:
                    continue
                visited[i] = True
                path.append(nums[i])
                dfs()
                path.pop()
                visited[i] = False

        dfs()
        return ans

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        ans = []
        path = []
        visited = [False] * len(nums)

        def dfs():
            if len(path) == len(nums):
                ans.append(path[:])
                return
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                    continue
                if visited[i]:
                    continue
                visited[i] = True
                path.append(nums[i])
                dfs()
                path.pop()
                visited[i] = False

        dfs()
        return ans

    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

    def canJump(self, nums: List[int]) -> bool:
        dp = [False] * len(nums)
        dp[0] = True
        jump = nums[0]
        for i in range(len(nums)):
            if jump >= i:
                dp[i] = True
                jump = max(jump, nums[i] + i)
        return dp[-1]

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        res = [intervals[0]]
        for interval in intervals[1:]:
            if res[-1][1] >= interval[0]:
                res[-1][1] = max(interval[1], res[-1][1])
            else:
                res.append(interval)
        return res

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        i = 0
        n = len(intervals)

        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1

        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i += 1
        res.append(newInterval)

        while i < n and intervals[i][0] > newInterval[1]:
            res.append(intervals[i])
            i += 1

        return res

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        if obstacleGrid[0][0] == 0:
            dp[0][0] = 1
        for i in range(1, m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = dp[i - 1][0]
        for j in range(1, n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = dp[0][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def search(self, nums: List[int], target: int) -> bool:
        left = 0
        right = len(nums)
        while left < right:
            mid = (left + right) // 2
            if target == nums[mid]:
                return True
            if nums[left] == nums[mid]:
                left += 1
                continue
            if nums[left] < nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid
                else:
                    left = mid + 1
            else:
                if nums[mid] <= target < nums[right - 1]:
                    left = mid + 1
                else:
                    right = mid
        return False

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        for i in range(1, len(triangle)):
            for j in range(len(triangle[i])):
                if j == 0:
                    triangle[i][j] += triangle[i - 1][j]
                if j == len(triangle[i]) - 1:
                    triangle[i][j] += triangle[i - 1][j - 1]
                else:
                    triangle[i][j] += min(triangle[i - 1][j], triangle[i - 1][j - 1])
        return min(triangle[-1])

    def solve(self, board: List[List[str]]) -> None:
        '''给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' 组成，捕获 所有 被围绕的区域：
            连接：一个单元格与水平或垂直方向上相邻的单元格连接。
            区域：连接所有 'O' 的单元格来形成一个区域。
            围绕：如果您可以用 'X' 单元格 连接这个区域，并且区域中没有任何单元格位于 board 边缘，则该区域被 'X' 单元格围绕。
            通过将输入矩阵 board 中的所有 'O' 替换为 'X' 来 捕获被围绕的区域。'''
        m, n = len(board), len(board[0])

        def dfs(i, j):
            while not 0 <= i < m or not 0 <= j < n or board[i][j] != "O":
                return
            board[i][j] = "#"
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)

        for i in range(m):
            if board[i][0] == "O":
                dfs(i, 0)
            if board[i][n - 1] == "O":
                dfs(i, n - 1)

        for j in range(n):
            if board[0][j] == "O":
                dfs(0, j)
            if board[m - 1][j] == "O":
                dfs(m - 1, j)
        for i in range(m):
            print(board[i])

        for i in range(m):
            for j in range(n):
                if board[i][j] == "#":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        res = 0
        total = 0
        cur = 0
        for i in range(len(gas)):
            total += gas[i] - cost[i]
            cur += gas[i] - cost[i]
            if cur < 0:
                cur = 0
                res = i
        return res

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        memo = {}

        def dfs(s):
            if s in memo:
                return memo[s]
            if not s:
                return True
            for word in wordDict:
                if s.startswith(word) and dfs(s[len(word):]):
                    memo[s] = True
                    return True
            memo[s] = False
            return False

        return dfs(s)

    def maxProduct(self, nums: List[int]) -> int:
        maxres = nums[0]
        minres = nums[0]
        res = maxres
        for num in nums[1:]:
            if num < 0:
                maxres, minres = minres, maxres
            maxres = max(num, num * maxres)
            minres = min(num, num * minres)
            res = max(res, maxres)
        return res

    def rotate(self, nums: List[int], k: int) -> None:
        def rev(nums, start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

        n = len(nums) - 1
        k = k % n
        rev(nums, 0, n)

        rev(nums, 0, k - 1)

        rev(nums, k, n)

    def rob(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(dp[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]

    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        return heapq.heappop(heap)

    def findKthSmallest(self, nums, k):
        heapq.heapify(nums)
        while k - 1 > 0:
            k -= 1
            heapq.heappop(nums)
        return heapq.heappop(nums)

    def findKthSmallest2(self, nums, k):
        left, right = nums[0], nums[-1]

        while left < right:
            mid = left + (right - left) // 2
            count = 0

            for num in nums:
                if num <= mid:
                    count += 1

            if count < k:
                left = mid + 1
            else:
                right = mid

        return left

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # 岛屿的最大面积， 1是陆地0是水
        m, n = len(grid), len(grid[0])

        def dfs(i, j):
            if not 0 <= i < m or not 0 <= j < n or grid[i][j] == 0:
                return 0
            grid[i][j] = 0

            res = dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1) + 1
            return res

        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    res = max(res, dfs(i, j))
        return res

    def nextGreaterElements2(self, nums: List[int]) -> List[int]:
        # 给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。
        # 如果不存在，则输出 -1 。
        stack = []
        n = len(nums)
        res = [-1] * n
        for i in range(n * 2):
            while stack and nums[stack[-1] % n] < nums[i % n]:
                index = stack.pop()
                res[index % n] = nums[i % n]
            stack.append(i)
        return res

    def findKthLargest3(self, nums: List[int], k: int) -> int:
        heap = []
        for i in range(len(nums)):
            heapq.heappush(heap, nums[i])
            if len(heap) > k:
                heapq.heappop(heap)
        return heapq.heappop(heap)

if __name__ == '__main__':
    so = Solution()

    print(so.findKthLargest3([1, 2, 3, 4, 5], 2))

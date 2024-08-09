#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/22 16:31
# @Author  : MinisterYU
# @File    : 链表习题.py
from typing import List, Optional


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    def reverseList(self, head):
        # if not head or not head.next:
        #     return head
        # next = self.reverseList(head.next)
        # head.next.next = head
        # head.next = None
        # return next
        pre, cur = None, head
        while cur:
            # cur.next, pre, cur = pre, cur, cur.next
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
        return pre

        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        '''
        https://leetcode.cn/problems/merge-in-between-linked-lists/
        给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。
        请你将 list1 中下标从 a 到 b 的全部节点都删除，并将list2 接在被删除节点的位置。'''
        prea = list1
        for _ in range(a - 1):
            prea = prea.next

        preb = prea
        for _ in range(b - a + 2):
            preb = preb.next
        prea.next = list2
        prec = list2
        while prec and prec.next:
            prec = prec.next
        prec.next = preb
        return list1

    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        https://leetcode.cn/problems/remove-nodes-from-linked-list/description/
        给你一个链表的头节点 head 。移除每个右侧有一个更大数值的节点。 返回修改后链表的头节点 head 。
        输入：head = [5,2,13,3,8]
        输出：[13,8]
        解释：需要移除的节点是 5 ，2 和 3 。
        - 节点 13 在节点 5 右侧。
        - 节点 13 在节点 2 右侧。
        - 节点 8 在节点 3 右侧。'''

        head = self.reverseList(head)  # [8, 3, 13, 2, 5]
        cur = head
        while cur and cur.next:
            if cur.val > cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return self.reverseList(head)

        cur = head
        while cur.next:
            if cur.val > cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return self.reverseList(head)

    def sortLIst(self, nums):
        if len(nums) < 2:
            return nums
        slow, fast = 0, 0
        while fast < len(nums):
            fast += 2
            slow += 1

        left = self.sortLIst(nums[:slow])
        right = self.sortLIst(nums[slow:])

        return self.mergelist(left, right)

    def mergelist(self, list1, list2):
        res = []
        x, y = 0, 0
        while x < len(list1) and y < len(list2):
            if list1[x] < list2[y]:
                res.append(list1[x])
                x += 1
            else:
                res.append(list2[y])
                y += 1

        res.extend(list1[x:])
        res.extend(list2[y:])
        return res

    def selection(self, nums):
        n = len(nums)
        for i in range(n - 1):
            mid = i
            for j in range(i + 1, n):
                if nums[j] < nums[mid]:
                    mid = j
            nums[i], nums[mid] = nums[mid], nums[i]
        return nums

    def quickSort(self, nums):
        n = len(nums)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1]
        return nums

    def helper(self, queue):
        num = 0
        stack = []
        sign = "+"

        while queue:
            char = queue.popleft()
            if char.isdigit():
                num = num * 10 + (ord(char) - ord("0"))

            if char == "(":
                num = self.helper(queue)

            if char in "+-*/" or not queue:
                if sign == "+":
                    stack.append(num)
                if sign == "-":
                    stack.append(-num)
                if sign == "*":
                    stack[-1] *= num
                if sign == "/":
                    stack[-1] = int(stack[-1] / num)

                sign = char
                num = 0

            if char == ")":
                break

        return sum(stack)

    def maxSubArray(self, nums: List[int]) -> int:
        '''给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
        '''
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        pre = [1] * n
        for i in range(1, n):
            pre[i] = pre[i - 1] * nums[i]
        post = [1] * n
        for i in range(n - 2, -1, -1):
            post[i] = post[i + 1] * nums[i]

        res = [1] * n
        for i in range(n):
            res[i] = pre[i] * post[i]
        return res

    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        res = 0
        for i in range(len(s)):
            char = s[i]
            if char == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res

    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = grid[i][0] + dp[i - 1][0]
        for j in range(n):
            dp[0][j] = grid[0][j] + dp[0][j - 1]
        for i in range(1, m):
            for j in range(n):
                dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    def longestPalindrome(self, s: str) -> str:
        def isPalindrome(s, left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1: right]

        n = len(s)
        res = ''
        for i in range(n):
            res = max(res, isPalindrome(s, i, i), isPalindrome(s, i, i + 1), key=len)
        return res

    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        res = 0
        for i in range(len(s)):
            char = s[i]
            if char == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
            res = max(res, i - stack[-1])
        return res

    def canPartition(self, nums: List[int]) -> bool:
        sums = sum(nums)
        if sums % 2 == 1:
            return False
        targer = sums // 2
        dp = [0] * (targer + 1)
        for i in range(len(nums)):
            for j in range(targer, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
        return dp[-1] == targer

    def mergeAlternately(self, word1: str, word2: str) -> str:
        m, n = len(word1), len(word2)
        x = min(m, n)
        alter = True
        res = ''
        for i in range(x):
            res += word1[i] if alter else word2[i]
            alter = not alter
        res += word1[m:] if m > n else word2[n:]
        return res
    def reverseVowels(self, s: str) -> str:
        vowels = ['a', 'e', 'i', 'o', 'u']
        left = 0
        right = len(s) - 1
        s = list(s)
        while left < right:
            if s[left] not in vowels:
                left += 1
            elif s[right] not in vowels:
                right -= 1
            else:
                s[left], s[right] = s[right], s[left]
        return ''.join(s)



if __name__ == '__main__':
    so = Solution()
    nums = [3, 5, -1, 2, 4, 5]
    print(so.sortLIst(nums))
    print(so.selection(nums))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/9 14:31
# @Author  : MinisterYU
# @File    : temp_cases.py
import collections
from typing import List, Optional
from collections import defaultdict, Counter, deque, OrderedDict
import math
from 链表 import ListNode, TreeNode

'''
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。

 

示例 1:

输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
'''


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        counter_p = Counter(p)
        window = need = len(p)
        res = []
        for right in range(len(s)):
            char = s[right]
            if char in counter_p:
                if counter_p[char] > 0:
                    need -= 1
                counter_p[char] -= 1

            left = right - window
            if left >= 0:
                char = s[left]
                if char in counter_p:
                    if counter_p[char] <= 0:
                        need += 1
                    counter_p[char] += 1

            if need == 0:
                res.append(left + 1)

        return res

    def subarraySum(self, nums: List[int], k: int) -> int:
        '''
        给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。子数组是数组中元素的连续非空序列。
        示例 1：
        输入：nums = [1,1,1], k = 2
        输出：2
        示例 2：
        输入：nums = [1,2,3], k = 3
        输出：2
        '''

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        '''
         给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

         返回 滑动窗口中的最大值 。

         示例 1：

         输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
         输出：[3,3,5,5,6,7]
         解释：
         滑动窗口的位置                最大值
         ---------------               -----
         [1  3  -1] -3  5  3  6  7       3
          1 [3  -1  -3] 5  3  6  7       3
          1  3 [-1  -3  5] 3  6  7       5
          1  3  -1 [-3  5  3] 6  7       5
          1  3  -1  -3 [5  3  6] 7       6
          1  3  -1  -3  5 [3  6  7]      7
        '''
        res = []
        stack = collections.deque()
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            stack.append(i)

            if stack and i - stack[0] >= k:
                stack.popleft()

            if i - k + 1 >= 0:
                res.append(nums[stack[0]])
        return res

    def minWindow(self, s: str, t: str) -> str:
        '''
        示例 1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
        '''
        res = ''
        res_len = len(s) + 1
        need = len(t)
        left = 0
        counter_t = collections.Counter(t)
        for right in range(len(s)):
            char = s[right]
            if char in counter_t:
                if counter_t[char] > 0:
                    need -= 1
                counter_t[char] -= 1

            while need == 0:
                if res_len > right - left + 1:
                    res_len = right - left + 1
                    res = s[left:right + 1]
                char = s[left]
                if char in counter_t:
                    if counter_t[char] >= 0:
                        need += 1
                    counter_t[char] += 1
                left += 1
        return res

    def maxSubArray(self, nums: List[int]) -> int:
        '''
 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
 子数组是数组中的一个连续部分。
 示例 1：

 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
 输出：6
 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

        '''
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        '''
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

示例 1：

输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
        '''
        res = []
        intervals.sort(key=lambda x: x[0])
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            if res[-1][1] < intervals[i][0]:
                res.append(intervals[i])
            else:
                res[-1][0] = min(res[-1][0], intervals[i][0])
                res[-1][1] = max(res[-1][1], intervals[i][1])
        return res

    def rotate(self, nums: List[int], k: int) -> None:
        '''
        给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
示例 1:

输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
        '''

        def rev(nums, left, right):
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        k = k % len(nums)
        rev(nums, 0, len(nums) - 1)
        rev(nums, 0, k)
        rev(nums, k, len(nums) - 1)

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

请 不要使用除法，且在 O(n) 时间复杂度内完成此题。



示例 1:

输入: nums = [1,2,3,4]
输出: [24,12,8,6]

        '''
        n = len(nums)
        prefix = [1] * n
        for i in range(1, n):
            prefix[i] = prefix[i - 1] * nums[i - 1]
        post = [1] * n
        for i in range(n - 2, -1, -1):
            post[i] = post[i + 1] * nums[i + 1]
        res = [1] * n
        for i in range(n):
            res[i] = prefix[i] * post[i]

        return res

    def firstMissingPositive(self, nums: List[int]) -> int:
        '''

给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。


示例 1：

输入：nums = [1,2,0]
输出：3
解释：范围 [1,2] 中的数字都在数组中。

先把小于等于零的全部置为正无穷大
再把大于数组长度的置为负数
再把小于零的值返回
        '''
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = math.inf

        for i in range(n):
            index = abs(nums[i])
            if index <= n:
                nums[index - 1] = -abs(nums[index - 1])

        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        last = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return last

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True

        def revHead(head):
            if not head or not head.next:
                return head
            last = revHead(head.next)
            head.next.next = head
            head.next = None
            return last

        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        last = revHead(slow)
        cur = head
        while last:
            if cur.val != last.val:
                return False
            cur = cur.next
            last = last.next
        return True

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def traverse(node):
            if not node:
                return 0
            left = traverse(node.left)
            right = traverse(node.right)
            return max(left, right) + 1

        return traverse(root)

    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        fast = slow = 1
        while fast < n:
            if nums[slow - 1] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow

    def removeDuplicatesII(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        fast = slow = 2
        while fast < n:
            if nums[slow - 2] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = [1] * n
        for i in range(1, n):
            prefix[i] = prefix[i - 1] * nums[i - 1]
        post = [1] * n
        for i in range(n - 2, -1, -1):
            post[i] = post[i + 1] * nums[i + 1]
        res = [1] * n
        for i in range(n):
            res[i] = prefix[i] * post[i]
        return res

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total = 0
        temp = 0
        res = 0
        for i in range(len(gas)):
            total += gas[i] - cost[i]
            temp += gas[i] - cost[i]
            if temp < 0:
                temp = 0
                res = i + 1
        return res if total >= 0 else -1

    def lengthOfLastWord(self, s: str) -> int:
        res = 0
        n = len(s)
        for i in range(n - 1, -1, -1):
            if s[i] == ' ' and res == 0:
                continue
            if s[i] == ' ' and res != 0:
                break
            if s[i] != ' ':
                res += 1

        return res

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) == 1:
            return strs[0]
        str0 = strs[0]
        left = 0
        right = len(strs)
        while left < right:
            mid = (right - left) // 2 + left
            if all(s.startswith(str0[:mid]) for s in strs[1:]):
                left = mid + 1
            else:
                right = mid
        return str0[:left]

    def reverseWords(self, s: str) -> str:
        n = len(s)
        left, right = 0, n - 1
        while left <= right and s[right].isspace():
            right -= 1
        while left <= right and s[left].isspace():
            left += 1
        res = []
        stack = []
        while left <= right:
            if s[left] == ' ' and stack:
                res.insert(0, ''.join(stack))
                stack.clear()
            elif not s[left].isspace():
                stack.append(s[left])
            left += 1
        return ' '.join(res)

    def strStr(self, haystack: str, needle: str) -> int:

        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i

        return -1

    def isPalindrome(self, s: str) -> bool:
        # left, right = 0, len(s) -1
        # while left <= right:
        #     if not s[left].isalpha():
        #         left += 1
        #     elif not s[right].isalpha():
        #         right -= 1
        #     elif s[left].lower() != s[right].lower():
        #         return False
        #     else:
        #         left += 1
        #         right -= 1
        # return True
        #
        new_s = ''.join(i for i in s if i.isalnum())
        n = len(new_s)
        l = n // 2
        return new_s[:l] == new_s[l + 1:][::-1] if n % 2 else new_s[:l] == new_s[l:][::-1]
        if n % 2:
            return new_s[:l] == new_s[l + 1:][::-1]
        else:
            return new_s[:l] == new_s[l:][::-1]


def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    if len(intervals) <= 1:
        return 0
    intervals.sort(key=lambda x: x[1])
    right = intervals[0][1]
    ans = 1
    print(intervals)
    for i in range(1, len(intervals)):
        if right <= intervals[i][0]:
            ans += 1
            right = intervals[i][1]
    return len(intervals) - ans


def kSmallestPairs(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    import heapq
    heap = []

    def push(i, j):
        if i < len(nums1) and j < len(nums2):
            heapq.heappush(heap, (nums1[i] + nums2[j], i, j))

    res = []
    push(0, 0)
    while len(res) < k:
        _, i, j = heapq.heappop(heap)
        res.append([i, j])
        push(i + 1, j)
        if j == 0:
            push(i, j + 1)
    return res


def numOfSubarrays(arr: List[int], k: int, threshold: int) -> int:
    '''
    给你一个整数数组 arr 和两个整数 k 和 threshold 。请你返回长度为 k 且平均值大于等于 threshold 的子数组数目。

    示例 1：
    输入：arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
    输出：3
    解释：子数组 [2,5,5],[5,5,5] 和 [5,5,8] 的平均值分别为 4，5 和 6 。其他长度为 3 的子数组的平均值都小于 4 （threshold 的值)。
    '''
    left = 0
    count = 0
    res = 0
    for right in range(len(arr)):
        count += arr[right]

        while right - left + 1 > k:
            count -= arr[left]
            left += 1
        if right - left + 1 == k and count >= threshold * k:
            res += 1
    return res


class Solutions2:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        counter = {}
        left = 0
        res = 0
        for right in range(n):
            counter[s[right]] = counter.get(s[right], 0) + 1

            while counter[s[right]] > 1:
                counter[s[left]] -= 1
                left += 1

            res = max(res, right - left + 1)
        return res

    def longestSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        res = 0
        count = 0
        for right in range(n):

            if nums[right] == 0:
                count += 1

            while count > 1:
                if nums[left] == 0:
                    count -= 1
                left += 1

            res = max(res, right - left + 1)

        return res - 1

    def longestSubstring(self, s: str, k: int) -> int:
        if len(s) < k:
            return 0
        counter = Counter(s)
        for char, count in counter.items():
            if count < k:
                sub_string_list = s.split(char)
                res = []
                for sub_string in sub_string_list:
                    res.append(self.longestSubstring(sub_string, k))
                return max(res)
        return len(s)


if __name__ == '__main__':
    so = Solutions2()
    # so.longestSubstring("abbbc", 2)
    nums = [2, 6, 7, 3, 1, 7]
    k = 3
    print(nums)
    print(nums[k - 1: ])
    for i, j in zip(nums, nums[k - 1:]):
        print(i, j)

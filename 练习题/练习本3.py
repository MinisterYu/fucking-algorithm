#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/9 上午11:24
# @Author  : MinisterYU
# @File    : 练习本3.py
import collections
from typing import List, Optional
from collections import defaultdict, deque, Counter
from math import inf, sqrt
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
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        '''
        给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
        https://leetcode.cn/problems/median-of-two-sorted-arrays/description/
        '''
        m, n = len(nums1), len(nums2)
        temp = []

        def dfs(i, j):
            if i == m:
                temp.extend(nums2[j:])
                return
            if j == n:
                temp.extend(nums1[i:])
                return

            if nums1[i] < nums2[j]:
                temp.append(nums1[i])
                dfs(i + 1, j)
            else:
                temp.append(nums2[j])
                dfs(i, j + 1)

        dfs(0, 0)
        if len(temp) % 2 == 1:
            return temp[len(temp) // 2]
        return (temp[len(temp) // 2] + temp[(len(temp)) // 2 - 1]) / 2

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # https://leetcode.cn/problems/3sum-closest/
        nums.sort()
        n = len(nums)
        min_diff = float('inf')
        res = 0
        for i in range(n - 2):
            j = i + 1
            k = n - 1
            while j < k:
                total = nums[i] + nums[j] + nums[k]
                if total == target:
                    return total

                diff = abs(total - target)
                if min_diff > diff:
                    min_diff = diff
                    res = total

                if total < target:
                    j += 1
                else:
                    k -= 1
        return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, n - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                k = j + 1
                l = n - 1
                while k < l:
                    total = nums[i] + nums[j] + nums[k] + nums[l]
                    if total == target:
                        res.append([nums[i], nums[j], nums[k], nums[l]])

                        k += 1
                        while k < l and nums[k] == nums[k - 1]:
                            k += 1
                        l -= 1
                        while k < l and nums[l] == nums[l + 1]:
                            l -= 1
                    elif total < target:
                        k += 1
                    else:
                        l -= 1
        return res

    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)

        while left < right:
            mid = (right - left) // 2 + left
            if nums[mid] == target:
                return mid

            # if nums[mid] < nums[-1]:
            #     if nums[mid] < target <= nums[-1]:
            #         left = mid + 1
            #     else:
            #         right = mid
            # else:
            #     if nums[0] <= target < nums[mid]:
            #         right = mid
            #     else:
            #         left = mid + 1

            if nums[mid] > nums[left]:
                if nums[left] <= target < nums[mid]:
                    right = mid
                else:
                    left = mid + 1

            else:
                if nums[mid] < target <= nums[right - 1]:
                    left = mid + 1
                else:
                    right = mid
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/
        def bisearch(nums, target):
            left = 0
            right = len(nums)
            while left < right:
                mid = (right - left) // 2 + left
                if nums[mid] == target:
                    return mid
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left

        begin = bisearch(nums, target)
        if begin == len(nums) or nums[begin] != target:
            return [-1, -1]
        end = bisearch(nums, target + 1) - 1
        return [begin, end]

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # https://leetcode.cn/problems/valid-sudoku/
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num != '.':
                    if num in rows[i]:
                        return False
                    rows[i].add(num)
                    if num in cols[j]:
                        return False
                    cols[j].add(num)
                    box_index = (i // 3) * 3 + j // 3
                    if num in boxes[box_index]:
                        return False
                    boxes[box_index].add(num)
        return True

    def calculate(self, s: str) -> int:
        que = deque(s)

        def help(que):
            stack = []
            num = 0
            sign = "+"
            while que:
                char = que.popleft()
                if char.isdigit():
                    num = num * 10 + int(char)

                elif char in "+-*/" or not que:
                    if sign == "+":
                        stack.append(num)
                    if sign == "-":
                        stack.append(-num)
                    if sign == "*":
                        stack[-1] = stack[-1] * num
                    if sign == "/":
                        stack[-1] = int(stack[-1] / num)

                    sign = char
                    num = 0
            return sum(stack)

        print(help(que))

    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = (right - left) // 2 + left
            if nums[mid] < nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]

    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        def bisearch(nums, target):
            left = 0
            right = len(nums)
            while left < right:
                mid = left + (right - left) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left

        n = len(intervals)

        # 将 索引加入到数组里面
        for k, v in enumerate(intervals):
            v.append(k)

        # 按照start 索引排序
        intervals.sort(key=lambda x: x[0])
        print(intervals)

        ans = [-1] * n
        for start, end, index in intervals:
            # 二分查找，如果没有找到，则返回
            i = bisearch(intervals, [end])

            if i < n:
                ans[index] = intervals[i][2]

        return ans

    def checkPossibility(self, nums: List[int]) -> bool:
        count = 0
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                count += 1

                if count > 1:
                    return False
                if i > 0 and nums[i - 1] > nums[i + 1]:
                    # 如果不是第一个元素， 且前序元素大于后续元素， 那么把后续元素变长 i
                    nums[i + 1] = nums[i]
                else:
                    # 当前元素加到跟后续元素一样大
                    nums[i] = nums[i + 1]
        return True

    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [n + 1] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1
            for j in range(i):
                if s[j: i] in dictionary:
                    dp[i] = min(dp[i], dp[j])
        return dp[-1]

    def sortColors(self, nums: List[int]) -> None:
        left = 0
        right = len(nums) - 1
        curr = 0
        while curr <= right:
            if nums[curr] == 0:
                nums[curr], nums[left] = nums[left], nums[curr]
                left += 1
                curr += 1
            elif nums[curr] == 2:
                nums[curr], nums[right] = nums[right], nums[curr]
                right -= 1
            else:
                curr += 1

    def minMoves(self, nums: List[int]) -> int:
        max_num = max(nums)
        count = 0
        for num in nums:
            count += max_num - num
        return count

    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        left = 0
        count = 0
        res = 0
        for right in range(len(arr)):
            count += arr[right]
            if right - left + 1 > k:
                count -= arr[left]
                left += 1
            if right - left + 1 == k and count > threshold * k:
                res += 1
        return res

    def splitArray(self, nums: List[int], k: int) -> int:
        '''
        https://leetcode.cn/problems/split-array-largest-sum/description/?envType=study-plan-v2&envId=2024-spring-sprint-100
        (「使……最大值尽可能小」是二分搜索题目常见的问法。)

        给定一个非负整数数组 nums 和一个整数 k ，你需要将这个数组分成 k 个非空的连续子数组。
        设计一个算法使得这 k 个子数组各自和的最大值最小。

        示例 1：

        输入：nums = [7,2,5,10,8], k = 2
        输出：18
        '''

        def check(target):
            cnt, total = 1, 0
            for num in nums:
                if total + num > target:
                    cnt += 1
                    total = num
                else:
                    total += num
            return cnt <= k

        left = max(nums)
        right = sum(nums)
        while left <= right:
            mid = (right - left) // 2 + left
            if check(mid):
                right = mid
            else:
                left = mid + 1
        return left

    def findRightInterval(self, intervals: List[List[int]]):
        for k, v in enumerate(intervals):
            v.append(k)
        intervals.sort(key=lambda x: x[0])
        # print(intervals)
        res = [-1] * len(intervals)
        for i in range(len(intervals)):
            start_i, end_i, index_i = intervals[i]
            for j in range(len(intervals)):
                start_j, end_j, index_j = intervals[j]
                if index_i == index_j:
                    continue
                if end_i <= start_j:
                    res[index_i] = index_j
        print(res)
        return res

    def equalPairs(self, grid: List[List[int]]) -> int:
        n = len(grid)
        cont = defaultdict(int)
        ans = 0
        for i in range(n):
            res = ''
            for j in range(n):
                res += f'#{grid[i][j]}'
            cont[res] += 1
        for j in range(n):
            res = ''
            for i in range(n):
                res += f'#{grid[i][j]}'
            if res in cont:
                ans += cont[res]
        return ans

    def findThePrefixCommonArray(self, a: List[int], b: List[int]) -> List[int]:
        ans = []
        const = defaultdict(int)
        for i in range(len(a)):
            const[a[i]] += 1
            const[b[i]] += 1
            ans.append(sum(i // 2 for i in const.values()))
        return ans

    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [n + 1] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1
            for j in range(i):
                if s[j: i] in dictionary:
                    dp[i] = min(dp[i], dp[j])
        return dp[-1]

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        res = n + 1
        left = 0
        cont = 0
        for right in range(n):
            cont += nums[right]
            while cont - nums[left] >= target:
                cont -= nums[left]
                res = min(res, right - left + 1)
                left += 1
        return res

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = 0
        cont = 1
        res = 0
        for right in range(n):
            cont *= nums[right]
            while cont // nums[left] >= k:
                cont //= nums[left]
                left += 1
            res += right - left + 1
        return res

    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = {0: 1}
        prefix = 0
        count = 0
        for num in nums:
            prefix += num
            target = prefix - k
            if target in prefix_sum:
                count += prefix_sum[target]
            prefix_sum[prefix] = prefix_sum.get(prefix, 0) + 1
        return count

    def lengthOfLongestSubstring(self, s: str) -> int:
        counter = defaultdict(int)
        res = 0
        left = 0
        for right in range(len(s)):
            counter[s[right]] += 1
            while counter[s[right]] > 1:
                counter[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res

    def longestPalindrome(self, s: str) -> str:
        def find(i, j, s):
            while i > 0 and j < len(s) and s[i] == s[j]:
                i -= 1
                j += 1

            return i + 1, j - 1

        res = 0
        ans = ''
        for i in range(len(s)):
            left, right = find(i, i, s)
            if right - left + 1 > res:
                res = right - left + 1
                ans = s[left: right]
            left, right = find(i, i + 1, s)
            if right - left + 1 > res:
                res = right - left + 1
                ans = s[left: right]
        return ans

    def longestCommonPrefix(self, strs: List[str]) -> str:
        str0 = strs[0]
        for i in range(len(strs[0])):
            char = str0[i]
            for substr in strs[1:]:
                if i >= len(substr) or substr[i] != char:
                    return substr[:i]
        return str0

    def generateParenthesis(self, n: int) -> List[str]:
        ans = []

        def back(left, right, path):
            if len(path) == 2 * n:
                ans.append(f'{path}')
                return

            if left < n:
                back(left + 1, right, path + '(')
            elif right < left:
                back(left, right + 1, path + ')')

        back(0, 0, '')
        return ans

    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        res = 0
        for i in range(len(s)):
            if s[i] == '(':
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
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[-1][-1]

    def isAdditiveNumber(self, num: str) -> bool:
        def check(num1, num2, substring):
            if not substring:
                return True

            subsums = str(int(num1) + int(num2))
            if substring.startswith(subsums):
                return check(num2, subsums, substring[len(subsums):])
            return False

        n = len(num)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                num1 = num[:i]
                num2 = num[i:j]
                if len(num1) > 0 and num1[0] == '0':
                    continue
                if len(num2) > 0 and num2[0] == '0':
                    continue
                if check(num1, num2, num[j:]):
                    return True
        return False

    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        visit = set()
        counter = Counter(s)
        for char in s:
            counter[char] -= 1
            if char in visit:
                continue
            while stack and stack[-1] > char and counter[stack[-1]] > 0:
                char = stack.pop()
                visit.remove(char)

            stack.append(char)
            visit.add(char)
        return ''.join(stack)

    def reverseVowels(self, s: str) -> str:
        vowels = ['a', 'e', 'i', 'o', 'u']
        s = list(s)
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] not in vowels:
                left += 1
            elif s[right] not in vowels:
                right -= 1
            else:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1
        return ''.join(s)

    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        res = 1
        left = 0
        count = 0
        for right in range(1, len(s)):
            if s[right] == s[right - 1]:
                count += 1

            if count > 1:
                left += 1
                while s[left] != s[left - 1]:
                    left += 1
                count -= 1

            res = max(res, right - left + 1)
        print(res)

    def maxSubArray(self, nums: List[int]) -> int:
        '''
        给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。子数组是数组中的一个连续部分。
        输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
        输出：6
        解释：连续子数组 [4,-1,2,1] 的和最大，为 6
        '''
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            if dp[i - 1] < 0:
                dp[i] = nums[i]
            else:
                dp[i] = nums[i] + dp[i - 1]
        return max(dp)

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        '''
        输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
        输出：[[1,6],[8,10],[15,18]]
        解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
        '''
        intervals.sort(key=lambda x: x[0])
        ans = [intervals[0]]
        for interval in intervals[1:]:
            if ans[-1][1] >= interval[0]:
                ans[-1][0] = min(ans[-1][0], interval[0])
                ans[-1][1] = max(ans[-1][1], interval[1])
            else:
                ans.append(interval)
        return ans

    def rotate(self, nums: List[int], k: int) -> None:
        '''
        输入: nums = [1,2,3,4,5,6,7], k = 3
        输出: [5,6,7,1,2,3,4]
        解释:
        向右轮转 1 步: [7,1,2,3,4,5,6]
        向右轮转 2 步: [6,7,1,2,3,4,5]
        向右轮转 3 步: [5,6,7,1,2,3,4]
        '''

        def reverse(left, right):
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        n = len(nums) - 1
        reverse(0, n)
        reverse(0, k - 1)
        reverse(k, n)

    def firstMissingPositive(self, nums: List[int]) -> int:
        '''
        输入：nums = [3,4,-1,1]
        输出：2
        解释：1 在数组中，但 2 没有。
        '''
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = inf

        for i in range(n):
            index = abs(nums[i])
            if index <= len(nums):
                nums[index - 1] = -abs(nums[index - 1])

        for i in range(n):
            if nums[i] > 0:
                return i + 1

        return n + 1

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(root, left, right):
            if not root:
                return True
            if root.val <= left or root.val >= right:
                return False
            l = dfs(root.left, left, root.val)
            r = dfs(root.right, root.val, right)
            return l and r

        return dfs(root, -inf, inf)

    def numIslands(self, grid: List[List[str]]) -> int:
        '''
        给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
        岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
        此外，你可以假设该网格的四条边均被水包围。
        '''
        m, n = len(grid), len(grid[0])

        def backtrack(grid, i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
                return
            grid[i][j] = '0'
            backtrack(grid, i + 1, j)
            backtrack(grid, i - 1, j)
            backtrack(grid, i, j - 1)
            backtrack(grid, i, j + 1)

        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    ans += 1
                    backtrack(grid, i, j)
        return ans

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        '''
        给你一个大小为 m x n 的二进制矩阵 grid 。
        岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
        岛屿的面积是岛上值为 1 的单元格的数目。
        计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
        '''
        m, n = len(grid), len(grid[0])

        def backtrack(grid, i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
                return 0
            grid[i][j] = 0
            count = backtrack(grid, i + 1, j) + backtrack(grid, i - 1, j) + backtrack(grid, i, j - 1) + backtrack(grid,
                                                                                                                  i,
                                                                                                                  j + 1) + 1
            return count

        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    ans = max(ans, backtrack(grid, i, j))
        return ans

    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]):
        # m, n = len(grid1), len(grid1[0])
        # self.grid1_list = []
        # self.grid2_list = []
        #
        # def dfs(grid, i, j, res):
        #     if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
        #         return
        #     res.append(f'{i}_{j}')
        #     grid[i][j] = 0
        #     dfs(grid, i + 1, j, res)
        #     dfs(grid, i - 1, j, res)
        #     dfs(grid, i, j + 1, res)
        #     dfs(grid, i, j - 1, res)
        #     return res
        #
        # for i in range(m):
        #     for j in range(n):
        #         if grid1[i][j] == 1:
        #             self.grid1_list.append(dfs(grid1, i, j, []))
        #         if grid2[i][j] == 1:
        #             self.grid2_list.append(dfs(grid2, i, j, []))
        # print(self.grid1_list)
        # print(self.grid2_list)
        # res = 0
        # for grid1_cell in self.grid1_list:
        #     for grid2_cell in self.grid2_list:
        #         if all(item in grid1_cell for item in grid2_cell):
        #             res += 1
        # print(res)
        # return res
        # 得到矩阵的行和列
        m, n = len(grid1), len(grid1[0])

        def dfs(grid, i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
                return
            grid[i][j] = 0
            dfs(grid, i + 1, j)
            dfs(grid, i - 1, j)
            dfs(grid, i, j + 1)
            dfs(grid, i, j - 1)

        # 首先将2中岛屿不被1中包含的剔除掉
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1 and grid1[i][j] == 0:
                    dfs(grid2, i, j)

        # 计算子岛屿数量
        res = 0
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1:
                    res += 1
                    dfs(grid2, i, j)
        return res

    def orangesRotting(self, grid: List[List[int]]) -> int:
        '''
在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：
值 0 代表空单元格；
值 1 代表新鲜橘子；
值 2 代表腐烂的橘子。
每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。

返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。
        '''
        m, n = len(grid), len(grid[0])
        pass

    def ddexist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])

        count_board = Counter([board[i][j] for i in range(m) for j in range(n)])
        count_word = Counter(word)
        for char in count_word:
            if count_word[char] > count_board[char]:
                return False

        len_board = m * n
        len_word = len(word)
        if len_word > len_board:
            return False

        # if count_word[word[-1]] > count_word[word[0]]:
        #     word = word[::-1]

        visited = [[False] * n for _ in range(m)]
        catch = {}

        def dfs(i, j, k):
            if k == len(word):
                return True

            if i < 0 or i >= m or j < 0 or j >= n or visited[i][j] or board[i][j] != word[k]:
                return False

            visited[i][j] = True
            res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
            visited[i][j] = False

        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    if dfs(i, j, 0):
                        return True
        return False

    def rob(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return max(nums)

        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]

    def maxProduct(self, nums: List[int]) -> int:
        max_result = [1] * len(nums)
        min_result = [1] * len(nums)
        max_result[0] = nums[0]
        min_result[0] = nums[0]
        res = -inf
        for i in range(1, len(nums)):
            if nums[i] < 0:
                max_result[i - 1], min_result[i - 1] = min_result[i - 1], max_result[i - 1]
            max_result[i] = max(nums[i], max_result[i - 1] * nums[i])
            min_result[i] = min(nums[i], min_result[i - 1] * nums[i])
            res = max(res, max_result[i])
        return res

    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [True] + [False] * n

        for i in range(1, n + 1):
            for j in range(i):
                if s[j: i] in wordDict and dp[j]:
                    dp[i] = True
        return dp[-1]

    def numSquares(self, n: int) -> int:
        nums = [i * i for i in range(1, int(sqrt(n)) + 1)]
        dp = [n + 1] * (n + 1)
        dp[0] = 0
        for i in range(len(nums)):
            for j in range(nums[i], n + 1):
                dp[j] = min(dp[j], dp[j - nums[i]] + 1)
        return dp[-1]

    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[-1][-1]

    def longestPalindrome(self, s: str) -> str:
        def palindrome(s: str, left: int, right: int) -> str:

            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1: right]

        if len(s) == 1:
            return s

        res = ''
        for i in range(len(s)):
            res = max(res, palindrome(s, i, i + 1), key=len)
            res = max(res, palindrome(s, i, i), key=len)
        print(res)
        return res

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    def minDistance(self, word1: str, word2: str) -> int:

        cache = {}

        def dfs(i, j):
            if (i, j) in cache:
                return cache[(i, j)]

            if len(word1) == i:
                return len(word2) - j

            if len(word2) == j:
                return len(word1) - i

            if word1[i] == word2[j]:
                res = dfs(i + 1, j + 1)
                cache[(i, j)] = res
                return res

            d = dfs(i + 1, j)
            s = dfs(i, j + 1)
            r = dfs(i + 1, j + 1)
            res = min(s, d, r) + 1
            cache[(i, j)] = res
            return res

        return dfs(0, 0)

    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)
        slow = fast = 2
        while fast < len(nums):
            if nums[slow - 2] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow

    def maxProfit(self, prices: List[int]) -> int:
        dp = [0, 0]
        dp[0] = 0  # 卖出
        dp[1] = -prices[0]  # 买入
        for price in prices:
            dp[0] = max(dp[0], dp[1] + price)
            dp[1] = max(dp[1], dp[0] - price)
        return max(dp)

    def canJump(self, nums: List[int]) -> bool:
        dp = [False] * len(nums)
        # dp[0] = True
        j = nums[0]
        for i in range(1, len(nums)):
            if j >= i:
                dp[i] = True
                j = max(j, nums[i] + i)
        return dp[-1]

    def jump(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        j = 0
        for i in range(1, len(nums)):
            while j + nums[j] < i:
                j += 1
            dp[i] = dp[j] + 1

        return dp[-1]

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_gas = 0
        current_gas = 0
        start_position = 0
        for i in range(len(gas)):
            total_gas += gas[i] - cost[i]
            current_gas += gas[i] - cost[i]
            if current_gas < 0:
                start_position = i + 1
                current_gas = 0
        return start_position if total_gas >= 0 else -1

    def singleNumber(self, nums: List[int]) -> int:
        cache = collections.defaultdict(int)
        for i in range(len(nums)):
            if nums[i] not in cache or cache[nums[i]] < 2:
                cache[nums[i]] += 1
            else:
                cache.pop(nums[i])
        print(cache)
        return list(cache.keys())[0]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        '''
        给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a ，b ，c ，使得 a + b + c = 0 ？请找出所有和为 0 且 不重复 的三元组。
        '''
        ans = []
        nums.sort()
        n = len(nums)
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j = i + 1
            k = n - 1
            while j < k:
                res = nums[i] + nums[j] + nums[k]
                if res > 0:
                    k -= 1
                elif res < 0:
                    j += 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return ans

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        res = n + 1
        left = 0
        total = 0
        for right in range(n):
            total += nums[right]
            while total >= target:
                res = min(res, right - left + 1)
                total -= nums[left]
                left += 1
        return res if res else 0

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        '''
        给定一个正整数数组 nums和整数 k ，请找出该数组内乘积小于 k 的连续的子数组的个数。
        nums = [10,5,2,6], k = 100
        输出: 8
        解释: 8 个乘积小于 100 的子数组分别为: [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]。
        需要注意的是 [10,5,2] 并不是乘积小于100的子数组。
        '''
        n = len(nums)
        if k < 1:
            return 0
        res = 0
        left = 0
        product = 1
        for right in range(n):
            product *= nums[right]
            while product >= k:
                product /= nums[left]
            res += right - left + 1
        return res

    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = 0
        total = 0
        res = 0
        for right in range(n):
            total += nums[right]
            while total >= k and left <= right:
                if total == 0:
                    res += 1
                total -= nums[left]
                left += 1
        return res

    def findMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        res = 0
        prefix = {0: -1}
        for i in range(n):
            if nums[i] == 0:
                count -= 1
            else:
                count += 1
            if count in prefix:
                res = max(res, i - prefix[count])
            else:
                prefix[count] = i

        return res

    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的某个变位词。
        换句话说，第一个字符串的排列之一是第二个字符串的 子串 。
        1 <= s1.length, s2.length <= 104
        '''
        if len(s1) > len(s2):
            return False

        count_s1 = Counter(s1)
        need = len(s1)
        left = 0
        for right in range(len(s2)):
            char = s2[right]
            if char in count_s1:
                if count_s1[char] > 0:
                    need -= 1
                count_s1[char] -= 1
            while need == 0:
                if right - left + 1 == len(s1):
                    return True
                char = s2[left]
                if char in count_s1:
                    if count_s1[char] >= 0:
                        need += 1
                    count_s1[char] += 1
                left += 1
        return False

    def lengthOfLongestSubstring(self, s: str) -> int:
        counter = defaultdict(int)
        left = 0
        res = 0
        for right in range(len(s)):
            counter[s[right]] += 1
            while counter[s[right]] > 1:
                counter[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res

    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        count = 0
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if i == j:
                    dp[i][j] = True
                    count += 1
                elif s[i] == s[j] and j - i == 1:
                    dp[i][j] = True
                    count += 1
                elif s[i] == s[j] and j - i > 1 and dp[i + 1][j - 1] == True:
                    dp[i][j] = True
                    count += 1
        return count

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        res = [0] * len(temperatures)

        for i in range(len(temperatures)):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                index = stack.pop()
                res[index] = i - index
            stack.append(i)
        return res

    def findBottomLeftValue(self, root: TreeNode) -> int:
        res = 0
        stack = deque([root])
        while stack:
            n = len(stack)
            for _ in range(n):
                node = stack.popleft()
                res = node.val
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        return res

    def largestValues(self, root: TreeNode) -> List[int]:
        res = []
        stack = deque([root])
        while stack:
            n = len(stack)
            temp_res = -inf
            for _ in range(n):
                node = stack.popleft()
                temp_res = max(temp_res, node.val)
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        return res

    def sumNumbers(self, root: TreeNode) -> int:
        self.res = 0

        def traverse(node, cur_num):
            if not node:
                return

            cur_num = cur_num * 10 + node.val
            if not node.left and not node.right:
                self.res += cur_num
            traverse(node.left, cur_num)
            traverse(node.right, cur_num)

            cur_num = (cur_num - node.val) // 10

        traverse(root, 0)
        return self.res

    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        prefix = {0: -1}
        self.res = 0

        def traverse(root, cur_sum):
            if not root:
                return
            cur_sum += root.val
            if cur_sum - targetSum in prefix:
                self.res += prefix[cur_sum - targetSum]
            prefix[cur_sum] = prefix.get(cur_sum, 0) + 1
            traverse(root.left, cur_sum)
            traverse(root.right, cur_sum)
            prefix[cur_sum] = prefix.get(cur_sum, 0) - 1

        traverse(root, 0)
        return self.res

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        '''给定一个整数数组 nums 和一个整数 k ，请返回其中出现频率前 k 高的元素。可以按 任意顺序 返回答案。'''

        heap = []
        count = Counter(nums)
        for key, val in count.items():
            heapq.heappush(heap, (val, key))
            if len(heap) > k:
                heapq.heappop(heap)
        return [i[1] for i in heap]

    def restoreIpAddresses(self, s: str) -> List[str]:
        ans = []
        path = []

        def dfs(index):
            if index == len(s) and len(path) == 4:
                ans.append('.'.join(path[:]))
                return
            if index == len(s) or len(path) == 4:
                return

            for i in range(index, min(index + 3, len(s))):
                if i > index and s[index] == '0':
                    break

                number = int(s[index: i + 1])
                if number <= 255:
                    path.append(str(number))
                    dfs(i + 1)
                    path.pop()

        dfs(0)
        return ans

    def minFlipsMonoIncr(self, s: str) -> int:
        n = len(s)
        dp_0 = [0] * (n + 1)
        dp_1 = [0] * (n + 1)

        for i in range(1, n + 1):
            if s[i - 1] == '0':
                dp_0[i] = dp_0[i - 1]
                dp_1[i] = dp_1[i - 1] + 1
            else:
                dp_0[i] = dp_0[i - 1] + 1
                dp_1[i] = min(dp_1[i - 1], dp_0[i - 1])
        return min(dp_0, dp_1)

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[-1][-1]

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        给定三个字符串 s1、s2、s3，请判断 s3 能不能由 s1 和 s2 交织（交错） 组成。

        两个字符串 s 和 t 交织 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：

        s = s1 + s2 + ... + sn
        t = t1 + t2 + ... + tm
        |n - m| <= 1
        交织 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
        提示：a + b 意味着字符串 a 和 b 连接。
        '''
        m, n, k = len(s1), len(s2), len(s3)
        if m + n != k:
            return False
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(m + 1):
            for j in range(n + 1):
                if i > 0 and s1[i - 1] == s3[ i + j - 1] and dp[i - 1][j]:
                    dp[i][j] = True
                if j > 0 and s2[j - 1] == s3[ i + j - 1] and dp[i][j - 1]:
                    dp[i][j] = True

        return dp[-1][-1]


if __name__ == '__main__':
    so = Solution()
    # print(so.threeSumClosest([-1, 2, 1, -4], 1))
    # s = []
    # s.reverse()
    # so.longestSemiRepetitiveSubstring("52233")

    res = so.longestCommonSubsequence('abcde', 'ace')
    print(res)

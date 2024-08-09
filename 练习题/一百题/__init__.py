#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 16:30
# @Author  : MinisterYU
# @File    : __init__.py.py
import heapq
from typing import List
import collections
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked
        has_map = collections.defaultdict(list)
        for i in range(len(nums)):
            has_map[nums[i]].append(i)

        keys = sorted(has_map.keys())
        left = 0
        right = len(keys) - 1
        while left <= right:
            if keys[left] + keys[right] > target:
                right -= 1
            elif keys[left] + keys[right] < target:
                left -= 1
            else:
                if keys[left] == keys[right] and len(has_map[keys[left]]) > 1:
                    return has_map[keys[left]][:2]
                elif left != right:
                    return has_map[keys[left]][0], has_map[keys[right]][0]
        return 0, 0

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        print(intervals)
        res = [intervals[0]]
        i = 1

        while i < len(intervals):
            if res[-1][1] < intervals[i][0]:
                res.append(intervals[i])
            else:
                res[-1][0] = min(res[-1][0], intervals[i][0])
                res[-1][1] = max(res[-1][1], intervals[i][1])
            i += 1

        print(res)

    def rotate(self, nums: List[int], k: int):
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n
        if k == 0:
            return nums
        nums = nums[n - k:] + nums[: n - k]
        nums.reverse()
        print(nums)

    def trans(self, s: str, n: int) -> str:
        if len(s) == 0:
            return s
        res = ""
        for i in range(len(s)):
            if s[i].isalpha():
                res += s[i].swapcase()
            else:
                # 空格直接复制
                res += s[i]
        print(res)
        # 单词反序
        res = list(res.split(' '))[::-1]
        print(res)
        res = ' '.join(res)
        print(res)

    def longestCommonPrefix(self, strs: List[str]) -> str:
        strs_len = len(strs)
        char_len = len(strs[0])

        # for char_index in range(char_len):
        #     for str_index in range(strs_len):
        #         if len(strs[str_index]) == char_index or strs[str_index][char_index] != strs[0][char_index]:
        #             return strs[0][:char_index]
        # return strs[0]
        for char_index in range(char_len):
            for str_index in range(strs_len):
                if len(strs[str_index]) == char_len or strs[str_index][char_index] != strs[0][char_index]:
                    return strs[0][:char_index]

        return strs[0]

    def lemonadeChange(self, bills: List[int]) -> bool:
        counter = collections.Counter()
        for bill in bills:
            if bill == 5:
                counter[5] += 1

            elif bill == 10:
                if not counter[5]:
                    return False
                counter[5] -= 1
                counter[10] += 1

            elif bill == 20:
                if counter[10]:
                    if not counter[5]:
                        return False
                    counter[5] -= 1
                    counter[10] -= 1

                elif counter[5]:
                    if counter[5] < 3:
                        return False
                    counter[5] -= 3
        print(counter)
        return True

    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        p1 = sorted(people, key=lambda x: (-x[0], x[1]))
        p2 = sorted(people, key=lambda x: (-x[0]))

        print(p1)
        print(p2)
        res = collections.deque()
        for p in p1:
            res.insert(p[1], p)
        print(list(res))

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.res = 0

        def bfs(i, j):
            queue = collections.deque()
            queue.append([i, j])
            grid[i][j] = 0
            res = 1
            while queue:
                x, y = queue.popleft()
                for dx, dy in dirs:
                    nx, ny = dx + x, dy + y
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                        res += 1
                        queue.append([nx, ny])
                        grid[nx][ny] = 0
            return res

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    self.res = max(self.res, bfs(i, j))
        return self.res

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                index = stack.pop()
                res[index] = nums[i]
            stack.append(i)
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                index = stack.pop()
                print(index)
                if res[index] == -1:
                    res[index] = nums[i]
            stack.append(i)

        print(res)

    def rotate2(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                if i == j:
                    continue
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        print(matrix)

    def plusOne(self, digits: List[int]) -> List[int]:
        digits = digits[::-1]
        res = []
        carry = 0

        for i in range(len(digits)):
            if i == 0:
                sum_ = digits[i] + 1 + carry
            else:
                sum_ = digits[i] + carry

            val = sum_ % 10
            carry = sum_ // 10
            res.insert(0, val)

        if carry:
            res.insert(0, carry)

    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (right - left) // 2 + left
            if nums[mid] < nums[-1]:
                right = mid
            else:
                left = mid + 1

        print(nums[left])

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]):
        m, n = len(nums1), len(nums2)
        merged = []

        def traverse(i, j):
            if i == m:
                merged.extend(nums2[j:])
                return
            if j == n:
                merged.extend(nums1[i:])
                return

            if nums1[i] < nums2[j]:
                merged.append(nums1[i])
                traverse(i + 1, j)
            else:
                merged.append(nums2[j])
                traverse(i, j + 1)

        traverse(0, 0)
        if (m + n) % 2:
            return (merged[(m + n) // 2] + merged[(m + n) // 2 + 1]) / 2
        return merged[(m + 2) / 2]

    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int):
        m, n = len(nums1), len(nums2)
        if not m or not n:
            return []
        heap = []
        res = []

        def push(i, j):
            if i < m and j < n:
                heapq.heappush(heap, [nums1[i] + nums2[j], i, j])

        push(0, 0)
        while heap and len(res) < k:
            _, i, j = heapq.heappop(heap)
            res.append([nums1[i], nums2[j]])
            # print(i, j + 1)
            push(i, j + 1)
            if j == 0:
                # print(i + 1, j)
                push(i + 1, j)
        print(res)

    def maxVowels(self, s: str, k: int) -> int:
        candidates = ['a', 'e', 'i', 'o', 'u']
        left = 0
        max_count = 0
        cur_count = 0
        cur_len = 0
        for right in range(len(s)):
            cur_len += 1
            if s[right] in candidates:
                cur_count += 1

            if cur_len > k:
                if s[left] in candidates:
                    cur_count -= 1
                cur_len -= 1
                left += 1

            if cur_len == k:
                max_count = max(max_count, cur_count)
        return max_count

    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n

        slow = fast = 2
        while fast < len(nums):
            if nums[slow - 2] < nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        print(nums)
        return slow

    def evalRPN(self, tokens: List[str]) -> int:
        from operator import truediv
        res = 0
        stack = []
        while tokens:
            char = tokens.pop(0)
            if self.is_number(char):
                stack.append(int(char))
            else:
                num2 = stack.pop()
                num1 = stack.pop()
                if char == '+':
                    stack.append(num1 + num2)
                if char == '-':
                    stack.append(num1 - num2)
                if char == '*':
                    stack.append(num1 * num2)
                if char == '/':
                    stack.append((int(truediv(num1, num2))))

        print(stack)

    def is_number(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero = 0
        two = len(nums) - 1
        one = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                nums[one], nums[zero] = nums[zero], nums[one]
                one += 1
                zero += 1
            elif nums[i] == 2:
                nums[one], nums[two] = nums[two], nums[one]
                two -= 1
            else:
                one += 1

        zero = 0
        right = len(nums) - 1
        one = 0
        while one <= right:
            if nums[one] == 0:
                nums[one], nums[zero] = nums[zero], nums[one]
                one += 1
                zero += 1
            elif nums[one] == 2:  # 当前元素为蓝色
                nums[one], nums[right] = nums[right], nums[one]
                right -= 1
            else:
                one += 1



    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        nums.sort()
        min_dis = 0
        max_dis = nums[-1] - nums[0]
        while min_dis < max_dis:
            mid_dis = (max_dis + min_dis) // 2
            count = 0
            i = 0
            for j in range(len(nums)):
                while nums[j] - nums[i] > mid_dis:
                    i += 1
                count += j - i

            if count < k:
                min_dis = mid_dis + 1
            else:
                max_dis = mid_dis
        return min_dis

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.res = 0
        prefix = {0:1}

        def backtrack(root, curSum):
            if not root:
                return 0
            curSum += root.val
            if curSum - targetSum in prefix:
                self.res += prefix[curSum - targetSum]

            prefix[curSum] = prefix.get(curSum, 0) + 1
            backtrack(root.left, curSum)
            backtrack(root.right, curSum)
            prefix[curSum] = prefix.get(curSum, 0) - 1

        backtrack(root, 0)
        return self.res




if __name__ == '__main__':
    so = Solution()
    # so.lemonadeChange([5,5,5,20,5,10,5,20,20,5])
    # so.reconstructQueue([[9, 0], [7, 0], [1, 9], [3, 0], [2, 7], [5, 3], [6, 0], [3, 4], [6, 2], [5, 2]])
    # so.maxAreaOfIsland( [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]])
    # so.nextGreaterElements([1,2,3,4,3])
    # so.nextGreaterElements([5,4,3,2,1])
    # so.plusOne([9])
    # so.findMin([3,4,5,1,2])
    # so.kSmallestPairs([1, 2, 3, 4, 5], [1, 2, 3, 3, 4], 25)
    # [[1, 1], [1, 1], [1, 2], [2, 1], [1, 2], [2, 2], [1, 3], [1, 3], [2, 3]]
    # [[1, 1], [1, 1], [1, 2], [1, 2], [2, 1], [1, 3], [1, 3], [2, 2], [2, 3]]
    # so.removeDuplicates([1, 1, 1, 2, 3])
    # so.evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"])
    so.sortColors([0, 1, 2, 0, 1, 2])
    s = 4
    print(s.bit_count())
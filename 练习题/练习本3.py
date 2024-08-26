#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/9 上午11:24
# @Author  : MinisterYU
# @File    : 练习本3.py
from typing import List, Optional
from collections import defaultdict, deque, Counter


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


if __name__ == '__main__':
    so = Solution()
    # print(so.threeSumClosest([-1, 2, 1, -4], 1))
    # s = []
    # s.reverse()
    so.longestSemiRepetitiveSubstring("52233")

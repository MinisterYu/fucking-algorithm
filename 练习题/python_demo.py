# coding:utf-8
import collections
from typing import List
from collections import Counter
from 链表 import ListNode

import heapq


class Solution:

    def numTeams(self, rating: List[int]) -> int:
        res = 0
        n = len(rating)
        for i in range(1, n - 1):
            left_small, left_big = 0, 0
            right_small, right_big = 0, 0
            for j in range(i):
                if rating[j] < rating[i]:
                    left_small += 1
                else:
                    left_big += 1
            for j in range(i + 1, n):
                if rating[j] < rating[i]:
                    right_small += 1
                else:
                    right_big += 1

            res += left_small * right_big + left_big * right_small
        return res

    def lastStoneWeightII(self, stones: List[int]) -> int:
        cap = sum(stones) // 2
        dp = [0] * (cap + 1)
        for i in range(len(stones)):
            for j in range(cap, stones[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
        return sum(stones) - 2 * dp[-1]

    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return 0
        dp = [0] * len(nums)
        for i in range(2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i - 1] + 1
        return sum(dp)

    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        n = len(books)
        dp = [0] + [float('inf')] * n
        for i in range(1, n + 1):
            width = 0
            height = 0
            for j in range(i, 0, -1):
                width += books[j - 1][0]
                if width > 0:
                    break
                height = max(height, books[j - 1][1])
                dp[i] = min(dp[i], dp[i - 1] + height)
        return dp[-1]

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        '''
输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]

        '''
        intervals.sort(key=lambda x: x[0])
        n = len(intervals)
        i = 0
        res = []
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i += 1
        res.append(newInterval)
        while 1 < n and newInterval[1] < intervals[i][0]:
            res.append(intervals[i])
            i += 1

        return res

    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = ""

        for i in range(n - 1, -1, -1):  # 从下往上遍历
            for j in range(i, n):  # 从左往右遍历
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    # dp[i][j] 定义 s字符串中，从i到j的最长回文子串
                    # dp[i][j] = s[i] 等于 s[j] 且 （ 左右间隔小于2，（aa,aba） 或者上一个状态为一个回文串 ）
                    dp[i][j] = True
                if dp[i][j] and j - i + 1 > len(res):
                    res = s[i:j + 1]

        return res

    def longestPalindrome2(self, s):
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = ""

        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                if dp[i][j] and j - i + 1 > len(res):
                    res = s[i: j + 1]
        return res

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

    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[m + n + 1] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + 1
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + 1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        index_map = {}
        for i, v in enumerate(nums):
            res = target - v
            if res in index_map:
                return [i, index_map[res]]
            else:
                index_map[res] = i

    def longestConsecutive(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            cur_num = num
            cur_res = 1
            while cur_num + 1 in nums:
                cur_num += 1
                cur_res += 1
            res = max(res, cur_res)
        return res

    def maxArea(self, height: List[int]) -> int:
        res = 0
        left, right = 0, len(height) - 1
        while left < right:
            res = max(res, (right - left) * min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return res

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j = i + 1
            k = len(nums) - 1
            while j < k:
                s = nums[i] + nums[j] + nums[k]
                if s > 0:
                    k -= 1
                elif s < 0:
                    j += 1
                else:
                    res.append([nums[i], nums[j], nums[k]])
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
        return res

    def lengthOfLongestSubstring(self, s: str) -> int:
        '''
        输入: s = "abcabcbb"
        输出: 3
        解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
        '''
        if len(s) == 0:
            return 0
        counter = {}
        left = 0
        res = 0
        for right in range(len(s)):
            char = s[right]
            counter[char] = counter.get(char, 0) + 1
            while char in counter.keys() and counter[char] > 1:
                counter[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res

    def findAnagrams(self, s: str, p: str) -> List[int]:
        '''
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
        '''
        counter = Counter(p)
        need = len(p)
        res = []

        for right in range(len(s)):
            char = s[right]
            if char in counter.keys():
                if counter[char] > 0:
                    need -= 1
                counter[char] -= 1
            left = right - len(p)
            if left >= 0:
                char = s[left]
                if char in counter.keys():
                    if counter[char] >= 0:
                        need += 1
                    counter[char] += 1
            if need == 0:
                res.append(left + 1)

        return res

    def subarraySum(self, nums: List[int], k: int) -> int:
        '''

给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。

子数组是数组中元素的连续非空序列。

        '''
        res = 0
        prefix = {0: 1}
        prefix_sum = 0
        for i in range(len(nums)):
            prefix_sum += nums[i]
            if prefix_sum - k in prefix:
                res += prefix[prefix_sum - k]
            prefix[prefix_sum] = prefix.get(prefix_sum, 0) + 1
        print(prefix)
        return res

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        stack = collections.deque()
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            stack.append(i)

            if stack and i - stack[0] > k:
                stack.popleft()
            if i - k + 1 >= 0:
                res.append(nums[stack[0]])

        return res

    def minWindow(self, s: str, t: str) -> str:
        '''
        输入：s = "ADOBECODEBANC", t = "ABC"
        输出："BANC"
        '''
        counter = Counter(t)
        need = len(t)
        left = 0
        res = ''
        res_len = len(s) + 1
        for right in range(len(s)):
            char = s[right]
            if char in counter:
                if counter[char] > 0:
                    need -= 1
                counter[char] -= 1

            while need == 0:
                if res_len > right - left + 1:
                    res = s[left:right + 1]
                    res_len = right - left + 1
                char = s[left]
                if char in counter:
                    if counter[char] >= 0:
                        need += 1
                    counter[char] += 1
                left += 1

        return res

    def maxSubArray(self, nums: List[int]) -> int:
        '''
        输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
        输出：6
        解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
        '''
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]

        return max(dp)

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        '''
        输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
        输出：[[1,6],[8,10],[15,18]]
        解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
        '''
        intervals.sort(key=lambda x: x[0])
        res = []
        res.append(intervals[0])

        for interval in intervals[1:]:
            if interval[0] < res[-1][1]:
                res[-1][1] = max(res[-1][1], interval[1])
                res[-1][0] = min(res[-1][0], interval[0])
            else:
                res.append(interval)
        return res

    def rotate(self, nums: List[int], k: int) -> None:
        '''
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
        '''

        def revsers(nums, left, right):
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        k %= len(nums)
        revsers(nums, 0, len(nums) - 1)
        revsers(nums, 0, k - 1)
        revsers(nums, k, len(nums) - 1)

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
        '''
        prefix = [1] * len(nums)
        for i in range(1, len(nums)):
            prefix[i] = prefix[i - 1] * nums[i - 1]

        post = [1] * len(nums)
        for i in range(len(nums) - 2, -1, -1):
            post[i] = post[i + 1] * nums[i + 1]

        res = [1] * len(nums)
        for i in range(len(nums)):
            res[i] = prefix[i] * post[i]
        return res

    def minFlipsMonoIncr(self, s: str) -> int:
        '''
如果一个由 '0' 和 '1' 组成的字符串，是以一些 '0'（可能没有 '0'）后面跟着一些 '1'（也可能没有 '1'）的形式组成的，那么该字符串是 单调递增 的。

我们给出一个由字符 '0' 和 '1' 组成的字符串 s，我们可以将任何 '0' 翻转为 '1' 或者将 '1' 翻转为 '0'。

返回使 s 单调递增 的最小翻转次数。

示例 1：

输入：s = "00110"
输出：1
解释：我们翻转最后一位得到 00111.
        '''
        n = len(s)
        dp_0 = [0] * (n + 1)
        dp_1 = [0] * (n + 1)
        for i in range(1, n + 1):
            if s[i - 1] == '0':
                dp_0[i] = dp_0[i - 1]
                dp_1[i] = dp_1[i - 1] + 1
            else:
                dp_0[i] = dp_0[i - 1] + 1
                dp_1[i] = min(dp_0[i - 1], dp_1[i - 1])
        print(dp_0)
        print(dp_1)
        return min(dp_0[-1], dp_1[-1])

    def countSubstrings(self, s: str) -> int:
        '''
        给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。回文字符串 是正着读和倒过来读一样的字符串。子字符串 是字符串中的由连续字符组成的一个序列。
        示例 1：

        输入：s = "abc"
        输出：3
        解释：三个回文子串: "a", "b", "c"
        '''
        n = len(s)
        count = 0
        dp = [[False] * n for _ in range(n)]  # dp[i][j] 为从i到j为回文子串
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if i == j:
                    dp[i][j] = True
                    count += 1
                if s[i] == s[j] and j - i == 1:
                    dp[i][j] = True
                    count += 1
                if s[i] == s[j] and j - i > 1 and dp[i + 1][j - 1] == True:
                    dp[i][j] = True
                    count += 1
        return count

    def longestValidParentheses(self, s: str) -> int:
        # https://leetcode.cn/problems/longest-valid-parentheses/
        '''
        示例 1：
        输入：s = "(()"
        输出：2
        解释：最长有效括号子串是 "()"

        示例 2：
        输入：s = ")()())"
        输出：4
        解释：最长有效括号子串是 "()()"
        '''
        n = len(s)
        dp = [0] * n

        for i in range(1, n):
            if s[i] == "(":
                dp[i] = 0
            elif s[i] == ")":
                if s[i - 1] == "(":
                    dp[i] = dp[i - 2] + 2

                elif s[i - 1] == ")" and i - dp[i - 1] > 0 and s[(i - dp[i - 1]) - 1] == "(":
                    dp[i] = dp[i - 1] + 2

                    if i - dp[i - 1] - 2 >= 0:
                        dp[i] = dp[i] + dp[i - dp[i - 1] - 2]

        return max(dp)

    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * n for _ in range(m)]
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j + 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1][i - 1][j - 1]) + 1
        return dp[-1][-1]

    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_product = nums[0]
        min_product = nums[0]
        res = max_product
        for i in range(1, len(nums)):
            if nums[i] < 0:
                min_product, max_product = max_product, min_product
            max_product = max(max_product, max_product * nums[i])
            min_product = min(min_product, min_product * nums[i])
            res = max(res, max_product)
        return res

    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] = max(dp[i], (i - j) * j, (i - j) * dp[j])
        return max(dp)

    def minCut(self, s: str) -> int:
        # n = len(s)
        # dp = [[True] * n for _ in range(n)]
        # for i in range(n - 1, -1, -1):
        #     for j in range(i + 1, n):
        #         dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]
        #
        # dp_f = [n] * n
        # for i in range(n):
        #     if dp[0][i]:
        #         dp_f[i] = 0
        #     else:
        #         for j in range(i):
        #             if dp[j + 1][i]:
        #                 dp_f[i] = min(dp_f[i], dp_f[j] + 1)
        #
        # return dp_f[n - 1]
        n = len(s)
        dp = [0] * n
        is_palindrome = [[False] * n for _ in range(n)]

        for i in range(n):
            min_cut = i
            for j in range(i + 1):
                if i == j or (s[i] == s[j] and (i - j <= 1 or is_palindrome[j + 1][i - 1])):
                    is_palindrome[j][i] = True
                    if j == 0:
                        min_cut = 0
                    else:
                        min_cut = min(min_cut, dp[j - 1] + 1)
            dp[i] = min_cut

        return dp[n - 1]

    def longestConsecutive(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        for i in range(n):
            cur_res = 1
            cur_num = 0
            while cur_num + 1 in nums:
                cur_num += 1
                cur_res += 1
            res = max(res, cur_res)
        return res

    def maxSubArray(self, nums: List[int]) -> int:
        '''
        给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
        '''
        n = len(nums)
        dp = [0 for _ in range(n)]
        dp[0] = nums[0]
        for i in range(1, n):
            if dp[i - 1] >= 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

    def longestPalindrome_连续(self, s):
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = ""
        for i in range(n - 1, -1, -1):
            for j in range(i):
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                if dp[i][j] and j - i + 1 > len(res):
                    res = s[i: j + 1]
        return res

    def reverseString(self, s: List[str]) -> None:
        left = 0
        right = len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]

    def longestPalindrome(self, s: str) -> str:
        # TODO 最长回文子串
        def help(s, left, right):
            while left >= 0 and right < len(s) - 1 and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1: right]

        res = ""
        for i in range(len(s)):
            res_1 = help(s, i, i)
            res_2 = help(s, i, i + 1)
            res = max(res_1, res_2, res, key=len)
        return res

    def letterCasePermutation(self, s: str) -> List[str]:
        ans = []

        def backtrack(index, path):
            if index == len(s):
                ans.append(''.join(path[:]))
                return
            if s[index].isalpha():
                backtrack(index + 1, path.append(s[index].upper()))
                backtrack(index + 1, path.append(s[index].lower()))
            else:
                backtrack(index + 1, path.append(s[index]))

        backtrack(0, [])
        return ans

    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        seen = set()

        if len(word) > m * n:
            return False

        count = Counter(sum(board, []))
        count1 = [i for s in board for i in s]
        print(count1)
        for c, countWord in Counter(word).items():
            if count[c] < countWord:
                return False

        if count[word[0]] > count[word[-1]]:
            word = word[::-1]

        def backtrack(i, j, k):
            if k == len(word):
                return True

            if (
                    i < 0
                    or i >= m
                    or j < 0
                    or j >= n
                    or board[i][j] != word[k]
                    or (i, j) in seen
            ):
                return False

            seen.add((i, j))
            found = (
                    backtrack(i - 1, j, k + 1)
                    or backtrack(i + 1, j, k + 1)
                    or backtrack(i, j - 1, k + 1)
                    or backtrack(i, j + 1, k + 1)
            )
            seen.remove((i, j))

            return found

        for i in range(m):
            for j in range(n):
                if backtrack(i, j, 0):
                    return True
        return False

    # todo >= target lower_bound(nums, target)
    # todo >  target lower_bound(nums, target +1 )
    # todo <  target lower_bound(nums, target) -1
    # todo =< target lower_bound(nums, target+1) -1
    def lower_bound(self, nums, target):
        left = 0
        right = len(nums)
        while left < right:
            mid = (right - left) // 2 + left
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left


def longest_common_prefix(strs):
    # str0 = strs[0]
    # for i in range(len(str0)):
    #     char = str0[i]
    #     for strx in strs[1:]:
    #         if len(strx) <= i or strx[i] != char:
    #             return str0[:i]
    # return str0
    str0 = strs[0]
    left = 0
    right = len(str0)
    while left < right:
        mid = (right - left) // 2 + left
        if all(strx.startswith(str0[:mid + 1]) for strx in strs):
            left = mid + 1
        else:
            right = mid
    return str0[:left]


def findkaishi(nums, target):
    '''
    给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

    如果数组中不存在目标值 target，返回 [-1, -1]。

    你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。



    示例 1：

    输入：nums = [5,7,7,8,8,10], target = 8
    输出：[3,4]
    '''

    def lowerbound(nums, target):
        left, right = 0, len(nums)
        while left < right:
            mid = (right - left) // 2 + left
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left

    left = lowerbound(nums, target)
    if nums[left] != target:
        return [-1, -1]
    right = lowerbound(nums, target + 1) - 1
    return [left, right]


def find2(nums, ):
    '''
峰值元素是指其值严格大于左右相邻值的元素。

给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞ 。
    '''
    left = 0
    right = len(nums)
    while left < right:
        mid = (right - left) // 2 + left
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left


def find3(nums):
    '''给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
示例 1：
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。'''
    left = 0
    right = len(nums)
    while left < right:
        mid = (right - left) // 2 + left
        if nums[mid] > nums[-1]:
            left = mid + 1
        else:
            right = mid
    return nums[left]


# def findKthLargest(nums, k):
def findKthSmallest(nums, k):
    # heap = []
    # for num in nums:
    #     heapq.heappush(heap, num)
    #     if len(heap) > k:
    #         heapq.heappop(heap)
    # return heap[0]
    k = len(nums) - k + 1
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heapq.heappop(heap)


def search(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums)

    while left < right:
        mid = (right - left) // 2 + left

        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
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

    # todo linkenode

    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
    给你一个链表的头节点 head ，该链表包含由 0 分隔开的一连串整数。链表的 开端 和 末尾 的节点都满足 Node.val == 0 。

    对于每两个相邻的 0 ，请你将它们之间的所有节点合并成一个节点，其值是所有已合并节点的值之和。然后将所有 0 移除，修改后的链表不应该含有任何 0 。

    返回修改后链表的头节点 head 。
        """
        cur = head.next
        res = ListNode(0)

        dummy = res
        count = 0
        while cur:
            if cur:
                count += cur.val
            else:
                dummy.next = ListNode(val=count)
                count = 0
                dummy = dummy.next
            cur = cur.next
        return res.next







if __name__ == '__main__':
    # print(findKthSmallest([3, 5, 7, 9], 1))

    so = Solution()
    # res = so.maxSubArray([-2, -1])
    # print(res)
    # res = so.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    # print(res)
    # res = so.longestPalindrome('babad')
    # print(res)
    # res = so.letterCasePermutation('a1B2')

    # nums = [1, 3, 5, 7, 9, 11]
    # tar = 7
    # '''res >= target lower_bound(nums, target)'''
    # res = so.lower_bound(nums, tar)
    # print(nums[res])
    #
    # tar = 5
    # '''res <= target lower_bound(nums, target + 1) - 1'''
    # res = so.lower_bound(nums, tar + 1) - 1
    # print(nums[res])
    #
    # tar = 8
    # '''res > target lower_bound(nums, target + 1)'''
    # res = so.lower_bound(nums, tar + 1)
    # print(nums[res])
    #
    # tar = 6
    # '''res < target lower_bound(nums, target) - 1'''
    # res = so.lower_bound(nums, tar) - 1
    # print(nums[res])
    #
    # # print(longest_common_prefix(['flower', 'flo', 'flower']))
    # print(findkaishi([5, 7, 7, 8, 8, 10], 8))
    # import  itertools
    # l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # print(list(itertools.accumulate(l)))

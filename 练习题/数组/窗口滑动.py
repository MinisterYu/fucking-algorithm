#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 17:09
# @Author  : MinisterYU
# @File    : 窗口滑动.py
# @URL https://leetcode.cn/problems/MPnaiL/solutions/1503490/by-flix-0h27
import collections
from typing import List


class Solution:

    # todo 解决求一个大字符串、数组中，是否包含子字符串、子序列的通用解法
    def minWindow(self, s: str, t: str) -> str:

        if len(t) > len(s):
            return ''

        cnt = collections.Counter(t)  # 哈希表：记录需要匹配到的各个元素的数目
        need = len(t)  # 记录需要匹配到的字符总数【need=0表示匹配到了】

        n = len(s)
        start, end = 0, -1  # 记录目标子串s[start, end]的起始和结尾
        min_len = n + 1  # 符合题意的最短子串长度【初始化为一个不可能的较大值】
        left = 0  # 滑动窗口的左右边界

        for right in range(n):

            # 窗口右边界右移一位
            ch = s[right]  # 窗口中新加入的字符
            if ch in cnt:  # 新加入的字符位于t中
                if cnt[ch] > 0:  # 对当前字符ch还有需求
                    need -= 1  # 此时新加入窗口中的ch对need有影响
                cnt[ch] -= 1

            # 窗口左边界持续右移
            while need == 0:  # need=0，当前窗口完全覆盖了t
                if right - left + 1 < min_len:  # 出现了更短的子串
                    min_len = right - left + 1
                    start, end = left, right

                ch = s[left]  # 窗口中要滑出的字符
                if ch in cnt:  # 刚滑出的字符位于t中
                    if cnt[ch] >= 0:  # 对当前字符ch还有需求，或刚好无需求(其实此时只有=0的情况)
                        need += 1  # 此时滑出窗口的ch会对need有影响
                    cnt[ch] += 1
                left += 1  # 窗口左边界+1

        return s[start: end + 1]

    # todo 连续子序列
    def checkInclusion(self, s1: str, s2: str) -> bool:

        m, n = len(s1), len(s2)
        if m > n:
            return False

        cnt = collections.Counter(s1)  # 哈希表：记录需要匹配到的各个字符的数目
        need = m  # 记录需要匹配到的字符总数【need=0表示匹配到了】

        for right in range(n):

            # 窗口右边界
            ch = s2[right]  # 窗口中新加入的字符
            if ch in cnt:  # 新加入的字符ch位于s1中
                if cnt[ch] > 0:  # 此时新加入窗口中的字符ch对need有影响
                    need -= 1
                cnt[ch] -= 1

            # 窗口左边界
            left = right - m
            if left >= 0:
                ch = s2[left]
                if ch in cnt:  # 刚滑出的字符位于s1中
                    if cnt[ch] >= 0:  # 此时滑出窗口的字符ch对need有影响
                        need += 1
                    cnt[ch] += 1

            if need == 0:  # 找到了一个满足题意的窗口
                return True

        return False

    # todo 395. 至少有 K 个重复字符的最长子串
    def longestSubstring(self, s: str, k: int) -> int:
        # 终止条件： 字符串的长度小于K，则最长结果返回为0
        if len(s) < k:
            return 0
        '''
        函数使用字典 counter 统计字符串 s 中每个字符的出现次数。
        接下来，函数遍历字典 counter，找到第一个出现次数小于 k 的字符。
        将字符串 s 根据该字符进行分割，得到左右两个子串。
        然后，函数递归地对左右两个子串进行处理，分别找到满足条件的最长子串的长度
        '''
        counter = collections.Counter(s)
        for char, count in counter.items():
            if count < k:
                substrings = s.split(char)
                res = []
                for substring in substrings:
                    res.append(self.longestSubstring(substring, k))
                return max(res)
                # return max(self.longestSubstring(substring, k) for substring in s.split(char))

        return len(s)

    # todo 424. 替换后的最长重复字符
    def characterReplacement(self, s: str, k: int) -> int:
        n = len(s)
        max_count = 0  # 记录最大出现次数的字符次数
        max_length = 0  # 记录最长字符串的长度
        counter = [0] * 26  # 计数器
        left = 0  # 窗口边界
        ord_A = ord('A')
        for right in range(n):

            index = ord(s[right]) - ord_A
            counter[index] += 1

            # 更新当前最大字符出现数
            max_count = max(max_count, counter[index])

            # 窗口长度不满足了,需要滑动窗口了
            while right - left + 1 > max_count + k:
                counter[ord(s[left]) - ord_A] -= 1
                left += 1

            max_length = max(max_length, right - left + 1)

        return max_length

    # https://leetcode.cn/problems/fruit-into-baskets/
    def totalFruit(self, fruits: List[int]) -> int:
        # 求数组中，最多包含2个不同数字的最大连续子序列
        n = len(fruits)
        if n < 3:
            return n

        counter = collections.Counter()
        ans = 0
        left = 0
        for right, value in enumerate(fruits):
            counter[value] += 1
            while len(counter) > 2:
                counter[fruits[left]] -= 1
                if counter[fruits[left]] == 0:
                    counter.pop(fruits[left])
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        ans = 0
        n = len(nums)
        left1 = left2 = 0
        total1 = total2 = 0
        for right in range(n):
            total1 += nums[right]

            while left1 <= right and goal < total1:
                total1 -= nums[left1]
                left1 += 1

            total2 += nums[right]
            while left2 <= right and goal <= total2:
                total2 -= nums[left2]
                left2 += 1

            ans += left2 - left1

        print(ans)

    def findAnagrams(self, s: str, p: str) -> List[int]:
        # https://leetcode.cn/problems/find-all-anagrams-in-a-string/
        from collections import defaultdict
        res = []
        need = n = len(p)
        counter = defaultdict(int)
        for c in p:
            counter[c] += 1

        for right, char in enumerate(s):
            if char in counter:
                if counter[char] > 0:
                    need -= 1
                counter[char] -= 1

            left = right - n
            if left >= 0:
                char = s[left]
                if char in counter:
                    if counter[char] >= 0:
                        need += 1
                counter[char] += 1

            if need == 0:
                res.append(right - n + 1)
        return res

    def longestOnes(self, nums: List[int], k: int) -> int:
        # https://leetcode.cn/problems/max-consecutive-ones-iii/?envType=study-plan-v2&envId=leetcode-75
        # 1004. 最大连续1的个数 , 允许有K个0
        res = 0
        left = 0
        need = 0

        for right, num in enumerate(nums):
            if num == 0:
                need += 1

            while need > k:
                if nums[left] == 0:
                    need -= 1
                left += 1

            res = max(res, right - left + 1)
        return res

    def longestSubarray(self, nums: List[int]) -> int:
        # https://leetcode.cn/problems/longest-subarray-of-1s-after-deleting-one-element/?envType=study-plan-v2&envId=leetcode-75
        # 1493. 删掉一个元素以后全为 1 的最长子数组
        left = 0
        max_length = 0
        need = 0
        for right, num in enumerate(nums):
            if num == 0:
                need += 1

            while need > 1:
                if nums[left] == 0:
                    need -= 1
                left += 1

            max_length = max(max_length, right - left)

        return max_length

    def maxVowels(self, s: str, k: int) -> int:
        # https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/?envType=study-plan-v2&envId=leetcode-75
        # 1456. 定长子串中元音的最大数目
        candidates = ['a', 'e', 'i', 'o', 'u']
        max_count = 0
        cur_count = 0
        cur_len = 0
        left = 0
        for right in range(len(s)):
            cur_len += 1
            if s[right] in candidates:
                cur_count += 1

            while cur_len > k:
                if s[left] in candidates:
                    cur_count -= 1
                cur_len -= 1
                left += 1

            if cur_len == k:
                max_count = max(max_count, cur_count)
        return max_count


if __name__ == '__main__':
    so = Solution()
    # so.numSubarraysWithSum([0, 1, 0, 0], 0)
    so.findAnagrams("cbaebabacd", 'abc')
    import itertools
    s = itertools.accumulate([1,2,3,4])
    print(list(s))
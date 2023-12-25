#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 17:09
# @Author  : MinisterYU
# @File    : 窗口滑动.py
# @URL https://leetcode.cn/problems/MPnaiL/solutions/1503490/by-flix-0h27
import collections


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
        left = right = 0  # 滑动窗口的左右边界

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

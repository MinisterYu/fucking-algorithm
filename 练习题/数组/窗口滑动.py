#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 17:09
# @Author  : MinisterYU
# @File    : 窗口滑动.py
# @URL https://leetcode.cn/problems/MPnaiL/solutions/1503490/by-flix-0h27
import collections

class Solution:
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

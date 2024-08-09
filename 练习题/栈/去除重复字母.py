#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/23 20:32
# @Author  : MinisterYU
# @File    : 去除重复字母.py

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        '''
        https://leetcode.cn/problems/remove-duplicate-letters/description/?envType=study-plan-v2&envId=2024-spring-sprint-100
        给你一个字符串 s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 返回结果的字典序最小（要求不能打乱其他字符的相对位置）。
        示例 1：
        输入：s = "bcabc"
        输出："abc"
        '''
        stack = []
        used = set()
        counter = {}
        for i in s:
            counter[i] = counter.get(i, 0) + 1

        for i in s:
            counter[i] -= 1

            if i in used:
                continue

            while stack and stack[-1] > i and counter[stack[-1]] > 0:
                big_i = stack.pop()
                used.remove(big_i)

            stack.append(i)
            used.add(i)

        return ''.join(stack)
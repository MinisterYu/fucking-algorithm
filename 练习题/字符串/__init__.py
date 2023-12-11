#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 17:35
# @Author  : MinisterYU
# @File    : __init__.py

from collections import deque


# TODO 最长回文子串
def longestPalindrome(s):
    start, end = 0, 0
    for i in range(len(s)):
        single_left, single_right = find_max(s, i, i)
        double_left, double_right = find_max(s, i, i + 1)
        if single_right - single_left > end - start:
            start, end = single_left, single_right
        if double_right - double_left > end - start:
            start, end = double_left, double_right
    return s[start: end + 1]


def find_max(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return left + 1, right - 1


# 示例用法
s = "abcbbc"
result = longestPalindrome(s)
print(result)


# TODO 最长公共前缀
class Solution:
    def longestCommonPrefix(self, strs) -> str:

        # for i in range(len(strs[0])):
        #     char = strs[0][i]
        #     for s in strs[1:]:
        #         if i >= len(s) or char != s[i]:
        #             return strs[0][:i]
        # return strs[0]

        strs.sort()
        left = 0
        right = len(strs[0])
        while left < right:
            mid = (right - left) // 2 + left
            # 如果都包含，左移
            if all(s.startswith(strs[0][:mid + 1]) for s in strs[1:]):
                left = mid + 1
            else:
                right = mid

        return strs[0][:left]


# TODO 找到最后一个单词 | 给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中 最后一个 单词的长度。
def lengthOfLastWord(s):
    right = len(s) - 1

    while 0 <= right and s[right] == ' ':
        right -= 1

    left = right

    while 0 <= left and s[left] != ' ':
        left -= 1

    return right - left

    # length = 0
    # end = len(s) - 1
    #
    # # 从字符串末尾开始遍历
    # while end >= 0:
    #     # 如果当前字符不是空格，则增加长度
    #     if s[end] != ' ':
    #         length += 1
    #     # 如果当前字符是空格且长度大于0，则表示最后一个单词已经遍历完，直接返回长度
    #     elif length > 0:
    #         return length
    #     end -= 1
    #
    # return length


print(lengthOfLastWord(""))


# TODO 151，反转字符串中的单词， 比如 " the sky is blue "  -> "blue is sky the"
def reverseWords(s):
    left = 0
    right = len(s) - 1

    while left <= right and s[left] == ' ':
        left += 1

    while left <= right and s[right] == ' ':
        right -= 1

    que = []
    word = []
    while left <= right:
        if word and s[left] == ' ':
            que.insert(0, ''.join(word))
            word = []
        elif s[left] != ' ':
            word.append(s[left])
        left += 1

    que.insert(0, ''.join(word))
    return ' '.join(que)


# TODO 151，反转字符串中的单词， 整体反转
def reverseWords2(s):
    s = s.strip()

    def revs_str(start, end, char_list):
        while start <= end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1

    start = 0
    char_list = list(s)
    # print(''.join(char_list))

    revs_str(start, len(s) - 1, char_list)

    for end in range(len(char_list) - 1):
        if char_list[end] == ' ':
            revs_str(start, end - 1, char_list)
            start = end + 1

    revs_str(start, len(char_list) - 1, char_list)

    return ''.join(char_list)

    # def revers_str(start, end, char_list):
    #     while start < end:
    #         char_list[start], char_list[end] = char_list[end], char_list[start]
    #         start += 1
    #         end -= 1
    #
    # left, right = 0, len(s) - 1
    # while left <= right and s[left] == ' ':
    #     left += 1
    #
    # while left <= right and s[right] == ' ':
    #     right -= 1
    #
    # char_list = list(s[left:right + 1])
    #
    # start = 0
    # revers_str(start, len(char_list) - 1, char_list)
    #
    # for end in range(len(char_list)):
    #     if char_list[end] == ' ':
    #         revers_str(start, end - 1, char_list)
    #         start = end + 1
    # revers_str(start, len(char_list) - 1, char_list)
    #
    # return ''.join(char_list)


print(reverseWords2(" the sky is blue "))


# TODO 异位词检查 "anagram" "nagaram"

def isAnagram(s, t):
    if len(s) != len(t):  # 如果两个字符串长度不相等，则不可能是变位词
        return False

    count = [0] * 26  # 使用一个长度为26的列表来记录每个字母出现的次数

    # 遍历字符串s，统计每个字母出现的次数
    for char in s:
        count[ord(char) - ord('a')] += 1

    # 遍历字符串t，减去每个字母出现的次数
    for char in t:
        count[ord(char) - ord('a')] -= 1
        if count[ord(char) - ord('a')] < 0:  # 如果出现负数，则表示两个字符串不是变位词
            return False

    return True  # 如果遍历结束后没有返回False，则表示两个字符串是变位词


s = "anagram"
t = "nagaram"
isAnagram(s, t)


# todo 单次规律  "abba"  "cat dog dog cat"
def wordPattern(pattern, s):
    words = s.split()  # 将字符串s按空格分割成单词列表
    if len(pattern) != len(words):  # 如果规律和单词数量不相等，则不遵循相同的规律
        return False

    pattern_to_word = {}  # 用于存储规律到单词的映射关系
    word_to_pattern = {}  # 用于存储单词到规律的映射关系

    for i in range(len(pattern)):
        if pattern[i] not in pattern_to_word:  # 如果规律中的字母不在映射中，则添加映射关系
            pattern_to_word[pattern[i]] = words[i]
        else:  # 如果规律中的字母已经在映射中，则检查映射关系是否一致
            if pattern_to_word[pattern[i]] != words[i]:
                return False

        if words[i] not in word_to_pattern:  # 如果单词不在映射中，则添加映射关系
            word_to_pattern[words[i]] = pattern[i]
        else:  # 如果单词已经在映射中，则检查映射关系是否一致
            if word_to_pattern[words[i]] != pattern[i]:
                return False

    print(pattern_to_word)
    print(word_to_pattern)

    return True  # 如果遍历结束后没有返回False，则表示遵循相同的规律


# todo 找到第一个出现的字母,找到就返回下标，没有就返回-1
def findFirstAlpha(s):
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1

    for i in range(len(s)):
        if char_count[s[i]] == 1:
            return i
    return -1

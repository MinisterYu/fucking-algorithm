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


# TODO 验证回文串，最多删除一个字符
def validPalindrome(s):
    left = 0
    right = len(s) - 1
    while left <= right:
        if s[left] == s[right]:
            left += 1
            right -= 1
        else:
            return isPalindrome(s, left + 1, right) or isPalindrome(s, left, right - 1)
    return True


def isPalindrome(s, left, right):
    while left <= right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True


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


print(lengthOfLastWord(""))


# TODO  151，反转字符串中的单词， 比如 " the sky is blue "  -> "blue is sky the"
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


# TODO 151，反转字符串中的单词， the sky is blue -> blue is sky the
def reverseWords2(s):
    s = s.strip()

    def revs_str(start, end, char_list):
        while start <= end:
            char_list[start], char_list[end] = char_list[end], char_list[start]
            start += 1
            end -= 1

    start = 0
    char_list = list(s)
    revs_str(start, len(s) - 1, char_list)
    end = 0
    while end < len(char_list):
        # for end in range(len(char_list) - 1):
        if char_list[end] == ' ':
            revs_str(start, end - 1, char_list)
            start = end + 1
        end += 1

    revs_str(start, len(char_list) - 1, char_list)

    return ''.join(char_list)


#
print(reverseWords2(" the   sky is blue "))


# TODO 给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。
# TODO s = "abcdefg", k = 2 输出："bacdfeg"
def reverseStr(s, k):
    cs = list(s)
    n = len(s)
    l = 0
    while l < n:
        r = l + k - 1
        reverse(cs, l, min(r, n - 1))
        l += 2 * k
    return ''.join(cs)


def reverse(cs, l, r):
    while l < r:
        cs[l], cs[r] = cs[r], cs[l]
        l += 1
        r -= 1


# TODO  反转字符串：s = "Mr Ding" 输出："rM gniD"
def reverseWords(s):
    chars = list(s)  # 将字符串转换为字符数组
    start = 0
    end = 0
    while end < len(chars):
        if chars[end] == ' ':
            reverse(chars, start, end - 1)  # 反转单词
            start = end + 1  # 更新下一个单词的起始位置
        end += 1
    reverse(chars, start, end - 1)  # 反转最后一个单词
    return ''.join(chars)  # 将字符数组转换为字符串


def reverse(chars, start, end):
    while start < end:
        chars[start], chars[end] = chars[end], chars[start]  # 交换字符
        start += 1
        end -= 1


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


# TODO 单次规律  "abba"  "cat dog dog cat"
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


# TODO 同构字符串 abba xyyx
def isIsomorphic(s, t):
    s2t = {}
    t2s = {}
    for i in range(len(s)):
        x, y = s[i], t[i]
        if x in s2t.keys() and s2t[x] != y or \
                y in t2s.keys() and t2s[y] != x:
            return False

        s2t[x] = y
        t2s[y] = x
    # print s2t
    # print t2s
    return True


# TODO 找到第一个出现的字母,找到就返回下标，没有就返回-1
def findFirstAlpha(s):
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1

    for i in range(len(s)):
        if char_count[s[i]] == 1:
            return i
    return -1


# TODO 520. 检测大写字母
def detectCapitalUse(word):
    if word.upper() == word:
        return True
    if word.lower() == word:
        return True
    n = len(word)
    idx = 1
    if word[0].isupper():
        while idx < n and word[idx].islower():
            idx += 1
    return idx == n


# TODO 821. 字符的最短距离，给你一个字符串 s 和一个字符 c ，且 c 是 s 中出现过的字符。
# TODO 输入：s = "loveleetcode", c = "e" # 输出：[3,2,1,0,1,0,0,1,2,2,1,0]
def shortestToChar(s, c):
    n = len(s)
    answer = [float('inf')] * n
    for i in range(n):
        if s[i] == c:
            answer[i] = 0
    for i in range(1, n):
        answer[i] = min(answer[i], answer[i - 1] + 1)
    for i in range(n - 2, -1, -1):
        answer[i] = min(answer[i], answer[i + 1] + 1)
    return answer


# TODO 字符串排序 0ab1c2 0a1b2c 或者 covid2019 c2v0i1d9
def reformat(s):
    digits = [i for i in s if i.isdigit()]
    chars = [i for i in s if i.isalpha()]

    if abs(len(digits) - len(chars)) > 1:
        return ''

    result = []
    # char多的时候，char为第一个
    is_turn = len(chars) >= len(digits)

    while digits or chars:
        if is_turn:
            result.append(chars.pop())
        else:
            result.append(digits.pop())
        is_turn = not is_turn

    return ''.join(result)


print(reformat('abc123d'))


# TODO 字符串排序（没搞懂跳转逻辑） 0ab1c2 0a1b2c 或者 covid2019 c2v0i1d9
def reformat2(s):
    digits = sum(i.isdigit() for i in s)
    chars = len(s) - digits

    if abs(digits - chars) > 1:
        return ''

    more_chars = chars > digits
    s_list = list(s)
    j = 1
    for i in range(0, len(s_list), 2):
        if s_list[i].isalpha() != more_chars:
            while s_list[j].isalpha() != more_chars:
                j += 2
            s_list[i], s_list[j] = s_list[j], s_list[i]
    return ''.join(s_list)


print(reformat2('1abc23d'))


# TODO 1668 找到最大的重复子字符串 ，二分，然后找子数组是不是在字符串里面
def maxRepeating(word, sequence):
    l = 0
    r = len(sequence) // len(word)
    while l < r:
        mid = (r - l + 1) // 2 + l
        if word * mid in sequence:
            l = mid
        else:
            r = mid - 1
    return l


print(maxRepeating("ba", "bababac"))



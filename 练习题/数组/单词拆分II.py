#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 00:25
# @Author  : MinisterYU
# @File    : 单词拆分II.py
'''
给定一个字符串 s 和一个字符串字典 wordDict ，在字符串 s 中增加空格来构建一个句子，使得句子中所有的单词都在词典中。
以任意顺序 返回所有这些可能的句子。
注意：词典中的同一个单词可能在分段中被重复使用多次。
https://leetcode.cn/problems/word-break-ii/description/
'''


class Solution(object):
    def wordBreak(self, s, wordDict):
        word_set = set(wordDict)
        memo = {}
        # min_s = min(len(word) for word in wordDict)

        def backtrack(s):
            # 如果已经计算过当前字符串的结果，直接返回
            if s in memo:
                return memo[s]

            # 如果剩下字符比最短的单词还短，返回一个空列表
            # if s < min_s:
            #     return [[]]

            # 如果当前字符串为空，返回一个空列表
            if not s:
                return [[]]

            result = []
            for word in word_set:
                # 如果当前单词是当前字符串的前缀
                if s.startswith(word):
                    # 递归计算剩余部分的结果
                    rest_sentences = backtrack(s[len(word):])
                    # 将当前单词与剩余部分的结果组合成句子
                    for rest_sentence in rest_sentences:
                        result.append([word] + rest_sentence)

            # 将结果存储到备忘录中
            memo[s] = result
            print(memo)
            return result

        sentences = backtrack(s)
        # 将句子列表转换为字符串列表
        return [' '.join(sentence) for sentence in sentences]


if __name__ == '__main__':
    so = Solution()
    res = so.wordBreak("wordworddog", ["word", "dog", "cat"])
    print(res)
    s = ["word", "dog", "cat"]
    print(s[:-1])

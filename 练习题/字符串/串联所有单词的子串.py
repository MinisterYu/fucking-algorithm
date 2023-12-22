#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 13:49
# @Author  : MinisterYU
# @File    : 串联所有单词的子串.py
from typing import List
from collections import Counter

class Solution:
    '''
    输入：s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
    输出：[6,9,12]
    '''
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words:
            return []

        word_len = len(words[0])
        window_len = word_len * len(words)
        n = len(s)
        result = []

        # 统计 words 中每个单词的出现次数
        word_count = Counter(words)

        for i in range(n - window_len + 1):
            # 使用一个哈希表来记录当前窗口中每个单词的出现次数
            window_count = Counter()

            # 滑动窗口遍历
            for j in range(i, i + window_len, word_len):
                word = s[j:j + word_len]

                # 如果当前单词不在 words 中，或者在窗口中出现的次数已经超过了 words 中的次数，则窗口无效
                if word not in word_count or window_count[word] >= word_count[word]:
                    break

                window_count[word] += 1

            # 如果窗口中的单词与 words 中的单词完全匹配，则将窗口的起始索引加入结果列表
            if window_count == word_count:
                result.append(i)

        return result



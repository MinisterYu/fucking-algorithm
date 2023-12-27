# coding:utf-8
from typing import List
# from collections import defaultdict, deque
# from collections import Counter
# from functools import lru_cache
# from functools import reduce
# from itertools import accumulate
import bisect
import operator
import collections
import functools
import itertools


# TODO 身高排序
def reconstructQueue(people):
    # 对people进行排序，按照身高降序，ki升序
    people.sort(key=lambda x: (-x[0], x[1]))
    print(people)

    # 初始化一个空队列
    queue = []

    # 将每个人按照ki值插入到队列中
    for person in people:
        queue.insert(person[1], person)

    return queue


# 示例用法
# people = [[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
# result = reconstructQueue(people)
# print(result)

def isWinner(player1: List[int], player2: List[int]) -> int:
    s1, s2 = 0, 0
    pre_index1, pre_index2 = -3, -3
    for cur_index in range(len(player1)):
        s1 += player1[cur_index] * (2 if cur_index - pre_index1 <= 2 else 1)
        s2 += player2[cur_index] * (2 if cur_index - pre_index2 <= 2 else 1)

        pre_index1 = cur_index if player1[cur_index] == 10 else pre_index1
        pre_index2 = cur_index if player2[cur_index] == 10 else pre_index2

    return 1 if s1 > s2 else 2 if s1 < s2 else 0

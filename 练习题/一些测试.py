# coding:utf-8
from typing import List
from collections import defaultdict, deque
from collections import Counter
from functools import lru_cache

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
from typing import List
import bisect


def findRightInterval(intervals: List[List[int]]):
    n = len(intervals)
    ans = [-1] * n
    for index, interval in enumerate(intervals):
        interval.append(index)

    # print(intervals)
    intervals.sort(key=lambda x: x[0])
    print(intervals)

    for _, end, index in intervals:
        i = bisect.bisect_left(intervals, [end])
        if i < n:
            ans[index] = intervals[i][2]

    print(ans)


# findRightInterval([[1,4],[2,3],[3,4]])
# list1 = [[1, 3], [3, 5], [7, 9]]
#
# bisect.insort_right(list1, [2, 6])
# print(list1)
# nums = 'abcdefg'
from functools import reduce

nums = [1, 2, 3, 4]
# print(reduce(lambda x, y: abs(x - y), nums))
print(reduce(lambda x, y: f'{x}-{y}', nums))

from itertools import accumulate

nums = [1, 2, 3, 4, 5]
prefix_sum = list(accumulate(nums))  # 计算累积和

print(prefix_sum)  # 输出 [1, 3, 6, 10, 15]

from itertools import accumulate
import operator

n = 5
factorial = list(accumulate(range(1, n + 1), operator.mul))  # 计算阶乘的累积乘积

print(factorial)  # 输出 [1, 2, 6, 24, 120]

s = '  hello world  '
print(s.split())


def removeDuplicates(s: str) -> str:
    '''
    输入："abbaca"
    输出："ca"
    '''
    stack = []
    for char in s:
        while stack and stack[-1] == char:
            stack.pop()

        stack.append(char)
    return ''.join(stack)

removeDuplicates("abbaca")


import heapq

heap = []
heapq.heappush(heap, (5, 'A', 1))
heapq.heappush(heap, (2, 'B', 2))
heapq.heappush(heap, (7, 'C', 3))

print(heap)  # 输出: [(2, 'B'), (5, 'A'), (7, 'C')]

# 位操作

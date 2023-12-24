# coding:utf-8
from typing import List
from collections import defaultdict, deque


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
list1 = [[1, 3], [3, 5], [7, 9]]

bisect.insort_right(list1, [2, 6])
print(list1)
nums = 'abcdefg'
from functools import reduce

print(reduce(lambda x, y: f'{x}->{y}', nums))

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n = len(nums)
for i in range(n-1, -1, -1):
    for j in range(i, n):
        print(f'i:{i}, j:{i}')

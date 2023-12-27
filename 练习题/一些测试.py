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
from 练习题.链表 import to_list, to_link, ListNode


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

list1 = [1, 2, 3, 4, 5]
link1 = to_link(list1)

cur = ListNode()
curr = cur
curr_1 = cur
def travers_1(head):
    global curr, curr_1
    if not head:
        return
    curr_1.next = ListNode(val=head.val)
    curr_1 = curr_1.next
    travers_1(head.next)
    # curr.next = ListNode(val=head.val)
    # curr = curr.next

travers_1(link1)
print(to_list(cur.next))






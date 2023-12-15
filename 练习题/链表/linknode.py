#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 17:36
# @Author  : MinisterYU
# @File    : linknode.py
from 练习题.链表 import *


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 链表排序
# 并排序法, 先递归把链表拆成最小为1or2的子链表，然后左右比对合并成一个新的，再继续往上递归

def link_sort(head: ListNode):
    if not head or not head.next:
        return head

    fast, slow = head, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    mid = slow.next
    slow.next = None

    left = link_sort((head))
    right = link_sort(mid)

    return merge(left, right)


def merge(left: ListNode, right: ListNode):
    dummy = ListNode()
    cur = dummy
    while left and right:
        if left.val < right.val:
            cur.next = left
            left = left.next
        else:
            cur.next = right
            right = right.next
        cur = cur.next

    if left:
        cur.next = left
    else:
        cur.next = right

    return dummy.next


'''
给定单个链表的头 head ，使用 插入排序 对链表进行排序，并返回 排序后链表的头 。
插入排序 算法的步骤:
1、插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
2、每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
3、重复直到所有输入数据插入完为止。
对链表进行插入排序。

'''


def insertSort(head: ListNode):
    if not head or not head.next:
        return head

    dummy = ListNode(next=head)
    cur = dummy

    while cur and cur.next:
        if cur.val <= cur.next.val:
            cur = cur.next
        else:  # cur > cur.next
            pre = dummy  # 从头开始

            while pre.next.val < cur.next.val:  # 如果前序是升序的，则跳过
                pre = pre.next

            temp = cur.next  # 把 cur.next
            cur.next = temp.next
            temp.next = pre.next
            pre.next = temp

        return dummy.next


def roatoe(nums, k):
    n = len(nums)
    move = n - k % n
    nums = nums + nums
    nums[:] = nums[move :move + n]

    print(nums)


roatoe([1, 2, 3, 4, 5, 6, 7], 3)

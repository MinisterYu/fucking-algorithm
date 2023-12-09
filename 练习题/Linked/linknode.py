#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 17:36
# @Author  : MinisterYU
# @File    : linknode.py
from 练习题.Linked import *


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

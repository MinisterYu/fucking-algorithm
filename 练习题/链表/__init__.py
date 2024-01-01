#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 17:36
# @Author  : MinisterYU
# @File    : __init__.py.py
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def to_link(list_):
    head = ListNode(list_[0])
    p = head
    for i in range(1, len(list_)):
        p.next = ListNode(list_[i])
        p = p.next
    return head


def to_list(head):
    l = []
    while head:
        l.append(head.val)
        head = head.next
    return l  # [:: -1]


node1 = to_link([1, 2, 3, 4, 5, 6])
node2 = to_link([1, 2, 3, 4, 5])


def mid(node):
    print(to_list(node))
    fast1 = node
    slow1 = node

    while fast1 and fast1.next:
        fast1 = fast1.next.next
        slow1 = slow1.next
    print('fast1 and fast1.next')
    print(slow1.val)


# mid(node1)
# mid(node2)


def mid2(node):
    print(to_list(node))
    fast1 = node
    slow1 = node
    while fast1.next and fast1.next.next:
        fast1 = fast1.next.next
        slow1 = slow1.next
    print('fast1.next and fast1.next.next')
    print(slow1.val)

# mid2(node1)
# mid2(node2)

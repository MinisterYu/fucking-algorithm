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
    return l   #[:: -1]
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/27 17:51
# @Author  : MinisterYU
# @File    : 删除链表中的元素.py
from typing import Optional
from 练习题.链表 import ListNode


class Solution:
    # TODO 19. 删除链表的倒数第 N 个结点
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(next=head)
        fast = dummy
        slow = dummy
        # fast 先走N步
        for i in range(n):
            fast = fast.next
        # fast 到尾部的时候，slow整好到导数N步
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next

    # TODO 82. 删除排序链表中的重复元素，如果元素重复则全部删除 ，（11123）-> 23
    def deleteDuplicates(self, head):
        dummy = ListNode(next=head)
        cur = dummy
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                val = cur.next.val
                while cur.next and cur.next.val == val:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next

    # TODO 83. 删除排序链表中的重复元素， 删除后不重复 11123 -> 123
    def deleteDuplicates(self, head):
        cur = head
        while cur and cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next

        return head

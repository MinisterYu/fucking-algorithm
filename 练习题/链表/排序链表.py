#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/27 16:51
# @Author  : MinisterYU
# @File    : 排序链表.py
from typing import Optional
from 练习题.链表 import ListNode


class Solution:

    # TODO 143. 重排链表: 1，2，3，4，5 排列成 1，5，2，4，3
    def reorderList(self, head):

        fast = head
        slow = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next

        # 切割链表 head = 1，2，3； cur=4，5，6
        cur = slow.next
        slow.next = None
        node2 = None

        while cur:
            tmp = cur.next
            cur.next = node2
            node2 = cur
            cur = tmp

        # node2 是倒序后的链表了
        node1 = head
        self.merge(node1, node2)

    def merge(self, head1, head2):
        dummy = ListNode()
        cur = dummy
        while head1 and head2:
            cur.next = head1
            head1 = head1.next
            cur = cur.next

            cur.next = head2
            head2 = head2.next
            cur = cur.next

            cur.next = head1 if head1 else head2
        return dummy.next

    # TODO 148. 排序列表，中间点 + 递归 + 排序 + 合并
    # 先递归查分成2个2个的比较大小合并，然后再逐步比较并合并
    def sortList(self, head):
        # 只有一个节点，则返回
        if not head or not head.next:
            return head

        # 进入迭代的准备
        slow, fast = head, head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        # 切割链表 head = 1，2，3； mid=4，5，6
        mid = slow.next
        slow.next = None  # 断开链表

        # 进入迭代
        return self.merge_compare(self.sortList(head), self.sortList(mid))

    def merge_compare(self, head1, head2):
        dummy = ListNode()
        cur = dummy
        while head1 and head2:

            if head1.val < head2.val:
                cur.next = head1
                head1 = head1.next
                cur = cur.next
            else:
                cur.next = head2
                head2 = head2.next
                cur = cur.next

            cur.next = head1 if head1 else head2

        return dummy.next

    # TODO 328. 奇偶链表, 奇数节点的和偶数节点 相互穿插
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        even_head = head.next
        odd_cur, even_cur = head, even_head

        while even_cur and even_cur.next:
            odd_cur.next = even_cur.next
            odd_cur = odd_cur.next
            even_cur.next = odd_cur.next
            even_cur = even_cur.next

        odd_cur.next = even_head
        return head

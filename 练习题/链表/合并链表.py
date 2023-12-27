#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/27 16:28
# @Author  : MinisterYU
# @File    : 合并链表.py
from typing import Optional
from 练习题.链表 import ListNode


class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # https://leetcode.cn/problems/merge-two-sorted-lists/
        # TODO 合并两个有序链表, 递归实现
        if not list1:
            return list2
        if not list2:
            return list1

        if list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2

    def mergeTwoLists2(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # https://leetcode.cn/problems/merge-two-sorted-lists/
        # TODO 合并两个有序链表, 循环实现
        dummy = ListNode()
        cur = dummy
        while list1 and list2:
            if list1.val < list2.val:
                cur.next = ListNode(val=list1.val)
                list1 = list1.next
            else:
                cur.next = ListNode(val=list2.val)
                list2 = list2.next
            cur = cur.next
        if list1:
            cur.next = list1
        if list2:
            cur.next = list2
        return dummy.next

    def mergeKLists(self, lists):
        # https://leetcode.cn/problems/merge-k-sorted-lists/
        # TODO 合并K个有序链表, 比对下排序的归并排序
        if not lists:
            return []
        if len(lists) <= 1:
            return lists[0]

        mid = len(lists) // 2
        left = self.mergeKLists(lists[:mid])
        right = self.mergeKLists(lists[mid:])

        return self.mergeTwoLists(left, right)

    def partition(self, head, x):
        # https://leetcode.cn/problems/partition-list/description/
        # TODO 分隔链表, 给一个节点值x,小于x的排前面，大于的排后面, 循环写法
        dummy1 = ListNode()
        dummy2 = ListNode()
        cur1 = dummy1
        cur2 = dummy2

        cur = head
        while cur:
            if cur.val < x:
                cur1.next = cur
                cur1 = cur1.next
            else:
                cur2.next = cur
                cur2 = cur2.next

            cur = cur.next

        cur2.next = None
        cur1.next = dummy2.next
        return dummy1.next

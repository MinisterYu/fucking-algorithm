#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/1 23:11
# @Author  : MinisterYU
# @File    : 环形链表.py
from typing import Optional
from 练习题.链表 import ListNode


class Solution:

    def hasCycle(self, head: ListNode) -> ListNode:
        # TODO 环形链表
        # https://leetcode.cn/problems/linked-list-cycle/

        slow, fast = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # TODO 环形链表II
        # https://leetcode.cn/problems/linked-list-cycle-ii/
        slow, fast = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                slow = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return slow
        return None


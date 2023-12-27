#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/27 15:42
# @Author  : MinisterYU
# @File    : 翻转链表.py
from typing import Optional
from 练习题.链表 import ListNode


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # https://leetcode.cn/problems/reverse-linked-list/
        # todo 反转一个链表,递归写法
        if not head or not head.next:
            return head

        last = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return last

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # https://leetcode.cn/problems/swap-nodes-in-pairs/
        # todo: 两两翻转节点,递归写法
        if not head or not head.next:
            return head

        next_pair = head.next.next
        new_head = head.next
        new_head.next = head

        head.next = self.swapPairs(next_pair)
        return new_head

    def reverseList2(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # https://leetcode.cn/problems/reverse-linked-list/
        # todo 反转一个链表,循环写法
        if not head or not head.next:
            return head
        cur = head
        last = None
        while cur:
            next_ = cur.next
            cur.next = last
            last = cur
            cur = next_
        return last

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        # https://leetcode.cn/problems/reverse-linked-list-ii/
        # todo 反转链表中间的一段, 循环写法
        dummy = ListNode(next=head)
        node1 = dummy

        for _ in range(left - 1):  # 假设left = 1 的话，那就是从头开始反转，这里就跳过了
            node1 = node1.next

        rev_head = node1.next
        rev_cur = rev_head
        rev_last = None

        for _ in range(left, right + 1):  # 假设 这里是 left 2, right 4，那么就是right要取到值，才能是2，3，,4，所以要+1
            next = rev_cur.next
            rev_cur.next = rev_last
            rev_last = rev_cur
            rev_cur = next

        node1.next = rev_last
        rev_head.next = rev_cur

        return dummy.next

    def swapPairs2(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # https://leetcode.cn/problems/swap-nodes-in-pairs/
        # todo: 两两翻转节点,循环写法
        dummy = ListNode(next=head)
        count = 0
        cur = dummy.next
        while cur:
            cur = cur.next
            count += 1  # 先统计节点个数

        p0 = dummy
        k = 2
        while count >= k:
            count -= k

            last = None
            cur = p0.next
            for _ in range(k):
                next_ = cur.next
                cur.next = last
                last = cur
                cur = next_

            next_ = p0.next
            p0.next.next = cur
            p0.next = last
            p0 = next_

        return dummy.next

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # https://leetcode.cn/problems/reverse-nodes-in-k-group/
        # todo: k个一组翻转节点,循环写法
        dummy = ListNode(next=head)
        count = 0
        cur = dummy.next
        while cur:
            cur = cur.next
            count += 1  # 先统计节点个数

        p0 = dummy
        while count >= k:
            count -= k

            last = None
            cur = p0.next
            for _ in range(k):
                next_ = cur.next
                cur.next = last
                last = cur
                cur = next_

            next_ = p0.next
            p0.next.next = cur
            p0.next = last
            p0 = next_

        return dummy.next
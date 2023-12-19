#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/18 23:34
# @Author  : MinisterYU
# @File    : 两数相加.py
from 练习题.链表 import ListNode, to_list, to_link


class Solution:
    def addInList(self, head1, head2):
        # write code here
        rev_head1 = self.reverse(head1)
        rev_head2 = self.reverse(head2)

        dummy = ListNode(0)
        carry = 0
        cur = dummy

        while rev_head1 or rev_head2 or carry:
            val1 = rev_head1.val if rev_head1 else 0
            val2 = rev_head2.val if rev_head2 else 0

            res = val1 + val2 + carry
            val = res % 10
            carry = res // 10
            cur.next = ListNode(val=val)
            cur = cur.next

            if rev_head1:
                rev_head1 = rev_head1.next
            if rev_head2:
                rev_head2 = rev_head2.next

        if rev_head1:
            cur.next = rev_head1
        if rev_head2:
            cur.next = rev_head2

        res_head = dummy.next
        return self.reverse(res_head)

    def reverse(self, head):
        prev = None
        curr = head

        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node

        return prev

if __name__ == '__main__':
    solution = Solution()
    s1 = to_link([5,9,7,5,7,1,2,6,4,2,7,8,9,6,1,6,6,1,1,4,2,9,5,5,0,4,6,3,0,4,3,5,6,7,0,5,5,4,4,0])
    s2 = to_link([1,3,2,5,0,6,0,2,1,4,3,9,3,0,9,9,0,3,1,6,5,7,8,6,2,3,8,5,0,9,7,9,4,5,9,9,4,9,3,6])
    res = solution.addInList(s1, s2)
    print(to_list(res))

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


# TODO 19. 删除链表的倒数第 N 个结点
class Solution(object):
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

    # TODO 61. 旋转列表
    def rotateRight(self, head, k):
        count = 1
        cur = head
        while cur.next:
            cur = cur.next
            count += 1
        move = count - k % count

        cur.next = head
        for _ in range(move):
            cur = cur.next

        ret = cur.next
        cur.next = None
        return ret

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




    # TODO 环形链表
    def detectCycle(self, head: ListNode) -> ListNode:
        slow, fast = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast.val == slow.val:
                return fast
        return None

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

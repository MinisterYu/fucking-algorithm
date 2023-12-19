#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/19 11:17
# @Author  : MinisterYU
# @File    : 逆序对.py

def merge(l1, l2):
    res = []
    count = 0
    i, j = 0, 0
    while i < len(l1) and j < len(l2):
        if l1[i] <= l2[j]:
            res.append(l1[i])
            i += 1
        else:
            res.append(l2[j])
            '''
            这行代码 `count += len(left) - i` 的目的是在合并过程中统计逆序对的数量。让我们解释一下：
            在归并排序中，`left` 和 `right` 是已经排好序的两个子数组，而在合并过程中，
            我们比较 `left[i]` 和 `right[j]` 的大小，如果 `left[i]` 大于 `right[j]`，
            说明 `left[i]` 及其右侧的所有元素都大于 `right[j]`。
            由于 `left` 和 `right` 都是已经排好序的，所以逆序对的数量就是 `left` 中剩余的元素个数，即 `len(left) - i`。
            通过累加这个数量，我们就可以在归并排序的过程中计算整个数组的逆序对总数。
            这样的方法充分利用了归并排序的性质，使得算法的时间复杂度为 O(n log n)，其中 n 是数组的长度。
            '''
            count += len(l1) - i
            j += 1

    res.extend(l1[i:])
    res.extend(l2[j:])
    return res, count


def sort_mege(nums):
    # return condition
    if len(nums) <= 1:
        return nums, 0

    mid = len(nums) // 2

    # into Recursion
    sorted_left, left_count = sort_mege(nums[:mid])
    sorted_right, right_count = sort_mege(nums[mid:])

    # handle this level logic
    sorted_list = []
    found_count = 0
    i, j = 0, 0
    while i < len(sorted_left) and j < len(sorted_right):
        if sorted_left[i] <= sorted_right[j]:
            sorted_list.append(sorted_left[i])
            i += 1
        else:
            sorted_list.append(sorted_right[j])
            '''
            这行代码 `count += len(left) - i` 的目的是在合并过程中统计逆序对的数量。让我们解释一下：
            在归并排序中，`left` 和 `right` 是已经排好序的两个子数组，而在合并过程中，
            我们比较 `left[i]` 和 `right[j]` 的大小，如果 `left[i]` 大于 `right[j]`，
            说明 `left[i]` 及其右侧的所有元素都大于 `right[j]`。
            由于 `left` 和 `right` 都是已经排好序的，所以逆序对的数量就是 `left` 中剩余的元素个数，即 `len(left) - i`。
            通过累加这个数量，我们就可以在归并排序的过程中计算整个数组的逆序对总数。
            这样的方法充分利用了归并排序的性质，使得算法的时间复杂度为 O(n log n)，其中 n 是数组的长度。
            '''
            found_count += len(sorted_left) - i
            j += 1

    sorted_list.extend(sorted_left[i:])
    sorted_list.extend(sorted_right[j:])

    return sorted_list, found_count + left_count + right_count


nums = [2, 1, 4, 3, 1]
print(sort_mege(nums))

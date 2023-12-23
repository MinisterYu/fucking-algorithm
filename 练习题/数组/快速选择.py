# coding:utf-8

import random

'''
这个算法的核心思想是选择一个基准数（这里使用随机选择），然后根据比基准数大、等于、小于基准数的元素将数组划分为三个部分。
然后，通过比较 k 与这三个部分的大小关系，决定在哪个部分继续查找。
递归地在相应的部分中进行查找，直到找到第 k 小的元素。

需要注意的是，这个实现中的划分方式使用了列表推导式，而在实际应用中，可以采用更高效的原地划分方式。
此外，算法的性能取决于基准数的选择和数组的划分情况。
'''


def quick_select_big(nums, k):
    if not nums:
        return None

    # 随机选择基准数
    pivot = random.choice(nums)
    # 划分数组，得到大于、等于、小于基准数的三个部分
    larger = [x for x in nums if x > pivot]
    equal = [x for x in nums if x == pivot]
    smaller = [x for x in nums if x < pivot]

    if k <= len(larger):
        # 第 k 大元素在 big 中，递归划分
        return quick_select_big(larger, k)
    if k <= len(larger) + len(equal):
        # 第 k 大元素在 equal 中，直接返回 pivot
        return pivot
    # 第 k 大元素在 small 中，递归划分
    return quick_select_big(smaller, k - len(larger) - len(equal))


def quick_select_small(nums, k):
    if not nums:
        return None

    # 随机选择基准数
    pivot = random.choice(nums)
    # 划分数组，得到大于、等于、小于基准数的三个部分
    larger = [x for x in nums if x > pivot]
    equal = [x for x in nums if x == pivot]
    smaller = [x for x in nums if x < pivot]

    if k <= len(smaller):
        # 第 k 小的元素在小于基准数的部分，递归进行查找
        return quick_select_small(smaller, k)

    if k <= len(smaller) + len(equal):
        # 找到了第 k 小的元素，直接返回
        return pivot

    # 第 k 小的元素在大于基准数的部分，递归进行查找
    return quick_select_small(larger, k - len(smaller) - len(equal))


# 示例用法
nums = [1, 2, 3, 4, 5, 6, 7, 8]
k = 3
result = quick_select_small(nums, k)
print(f"The {k}rd smallest element is: {result}")

result = quick_select_big(nums, k)
print(f"The {k}rd biggest element is: {result}")

# coding:utf-8
from typing import List

'''
这个模板使用了一个栈来辅助解决 Next Greater Element 问题。它遍历输入数组 nums，并维护一个递减/递减的栈
def nextGreaterElement(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            index = stack.pop()
            result[index] = nums[i]
        stack.append(i)

    return result
'''


class Solution:

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # https://leetcode.cn/problems/daily-temperatures/
        res = [0] * len(temperatures)
        stack = []
        for i in range(len(temperatures)):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                index = stack.pop()
                res[index] = i - index
            stack.append(i)
        return res

    def finalPrices(self, prices: List[int]) -> List[int]:
        '''
        给你一个数组 prices ，其中 prices[i] 是商店里第 i 件商品的价格。

        商店里正在进行促销活动，如果你要买第 i 件商品，那么你可以得到与 prices[j] 相等的折扣，

        其中 j 是满足 j > i 且 prices[j] <= prices[i] 的 最小下标 ，如果没有满足条件的 j ，你将没有任何折扣。

        请你返回一个数组，数组中第 i 个元素是折扣后你购买商品 i 最终需要支付的价格。
        '''
        stack = [0]
        result = [0] * len(prices)

        for i in range(len(prices) - 1, -1, -1):
            while len(stack) > 1 and stack[-1] > prices[i]:
                stack.pop()
            result[i] = prices[i] - stack[-1]
            stack.append(prices[i])
        return result

    def next_greater_element_reversed(self, nums):
        # 倒叙查找, 存放的是值
        stack = []
        result = [-1] * len(nums)  # 存放答案的数组

        for i in range(len(nums) - 1, -1, -1):  # 倒着往栈里放
            # while stack and nums[stack[-1]] <= nums[i]:  # 判定个子高矮
            while stack and stack[-1] <= nums[i]:
                stack.pop()  # 矮个起开，反正也被挡着了...

            result[i] = stack[-1] if stack else -1  # 这个元素身后的第一个高个
            # stack.append(i)  # 进栈，接受之后的身高判定吧！
            stack.append(nums[i])

        return result

    def next_greater_element(self, nums):
        '''
        使用单调栈找到下一个更大元素的位置

        1、创建一个空栈 stack 用于保存数组元素的索引，以及一个初始化长度与输入数组相同的结果数组 result，初始值都设为 -1。
        2、遍历数组 nums，对于每个元素 nums[i]：
            - 如果栈非空且当前元素大于栈顶元素（nums[i] > nums[stack[-1]]），则说明找到了栈顶元素的下一个更大元素。
                -弹出栈顶元素，并将其下一个更大元素的位置设为当前元素的位置 i。
            - 当前元素入栈，等待找到它的下一个更大元素。
        3、返回结果数组 result，其中每个位置的值表示对应位置元素的下一个更大元素的位置。
        ----
        这个算法的核心思想是通过单调递减的栈来维护一个候选的下一个更大元素的位置。
        当新元素入栈时，通过比较栈顶元素和当前元素的大小关系，
        可以确定栈顶元素的下一个更大元素的位置。这样一步步遍历数组，就可以得到每个元素的下一个更大元素的位置
        '''
        stack = []  # 创建一个空栈，用于保存数组元素的索引
        result = [-1] * len(nums)  # 初始化结果数组，将每个位置的值设为-1

        for i in range(len(nums)):
            # 当栈非空且当前元素大于栈顶元素时，说明找到了栈顶元素的下一个更大元素
            while stack and nums[i] > nums[stack[-1]]:
                result[stack.pop()] = i  # 栈顶元素的下一个更大元素的位置是当前元素的位置

            stack.append(i)  # 当前元素入栈，等待找到它的下一个更大元素
        print(result)
        return result


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 08:14
# @Author  : MinisterYU
# @File    : 单调栈.py

def numSubmat_DP(mat):
    m, n = len(mat), len(mat[0])
    dp = [[0] * n for _ in range(m)]  # 创建一个二维动态规划数组，用于存储每个位置的最小宽度
    result = 0

    for i in range(m):
        for j in range(n):
            if mat[i][j] == 1:
                dp[i][j] = dp[i][j - 1] + 1 if j > 0 else 1  # 计算每个位置的最小宽度

                min_width = dp[i][j]
                for k in range(i, -1, -1):
                    min_width = min(min_width, dp[k][j])  # 更新最小宽度
                    result += min_width  # 累加子矩形的个数
    for i in dp:
        print(i)
    return result


numSubmat_DP([[1, 0, 1],
              [1, 1, 0],
              [1, 1, 0]])


def numSubmat(mat):
    # 给你一个 m x n 的二进制矩阵 mat ，请你返回有多少个 子矩形 的元素全部都是 1 。
    # https://leetcode.cn/problems/count-submatrices-with-all-ones/submissions/494184301/
    m, n = len(mat), len(mat[0])
    heights = [0] * n  # 存储每个位置上方连续的 1 的个数
    result = 0

    for i in range(m):
        # 更新每个位置上方连续的 1 的个数
        for j in range(n):
            if mat[i][j] == 1:
                heights[j] += 1
            else:
                heights[j] = 0

        # 使用单调栈计算以当前位置为右下角的子矩形的个数
        stack = []
        count = 0
        for j in range(n):
            # 维持单调递增栈
            while stack and heights[stack[-1]] >= heights[j]:
                stack.pop()
            stack.append(j)

            # 计算以当前位置为右下角的子矩形的个数
            if stack[0] == 0:
                count = heights[j] * (j + 1)  # 当前位置为第一列时，子矩形的个数为高度乘以宽度
            else:
                count = heights[j] * (j - stack[-2])  # 子矩形的宽度为当前位置与栈顶位置之间的距离

            result += count

    return result

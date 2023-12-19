#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/18 10:15
# @Author  : MinisterYU
# @File    : 累加数306.py
class Solution:

    # 112358, 1+1 = 2, 2 + 3 =5 , 3+5 = 8
    def isAdditiveNumber(self, num: str) -> bool:

        n = len(num)
        for i in range(1, n):
            for j in range(i + 1, n):
                num1, num2 = num[:i], num[i:j]
                if len(num1) > 1 and num1[0] == '0' \
                        or len(num2) > 1 and num2[0] == '0':
                    continue

                if self.check_valid(num1, num2, num[j:]):
                    return True
        return False

    def check_valid(self, num1, num2, string: str):
        if not string:
            return True
        sum_string = str(int(num1) + int(num2))

        if string.startswith(sum_string):
            return self.check_valid(num2, sum_string, string[len(sum_string):])

        return False


if __name__ == '__main__':
    solution = Solution()
    res = solution.isAdditiveNumber("101")

    print(res)

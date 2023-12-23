# coding:utf-8
def wiggleMaxLength(nums):
    if len(nums) < 2:
        return len(nums)

    count = 1  # 记录摆动序列的长度
    prev_diff = 0  # 记录前一个元素与当前元素的差值

    for i in range(1, len(nums)):
        diff = nums[i] - nums[i - 1]
        if (diff > 0 and prev_diff <= 0) or (diff < 0 and prev_diff >= 0):
            count += 1
            prev_diff = diff

    return count

# 示例用法
nums = [1,7,4,9,2,5]
result = wiggleMaxLength(nums)
print(result)

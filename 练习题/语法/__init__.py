#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/5 10:13
# @Author  : MinisterYU
# @File    : __init__.py.py
import threading
import os

def my_thread_function():
    try:
        # 在这里编写你的线程代码
        file_path = os.path.join(os.path.abspath('./records'), 'test.log')
        with open(file_path, 'a+') as f:
            f.write('helloworld')
        f.close()
    except Exception as e:
        # 在这里处理异常
        print("线程异常:{}".format(e))
        for k in dir(e):
            if not k.startswith('__'):
                print('k:{}, attr: {}'.format(k, getattr(e, k)))

# 创建线程对象
my_thread = threading.Thread(target=my_thread_function)

# 启动线程
my_thread.start()

# 等待线程结束
my_thread.join()
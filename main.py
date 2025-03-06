# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import os
from typing import List
class A:
    def __init__(self):
        self.a = 1
        self.b = 2

    @staticmethod
    def func( l:List[int]):
        print(l)

    @staticmethod
    def func2():
        print(1)

    class A_c:
        def __init__(self):
            self.a = 3
            self.b = 4

        def func(self):
            A.func([self.a, self.b])



a= A().A_c()
a.func()
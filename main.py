# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
class test:
    def __init__(self,a=0):
        self.a=a

    def __call__(self):
        b=1
        self.c = [b]


a=test(1)
b=[a]*3
b[2].a=2
print(a.a)
a()
print(a.c)

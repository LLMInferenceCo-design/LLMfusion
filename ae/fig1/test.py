import pandas as pd

# 定义数据
data = [
    [100.0, 12.3, 8.7, 9.2],
    [9.8, 11.2, 7.6, 8.3]
]

# 创建多级列索引
columns = pd.MultiIndex.from_product([
    ['Lin1', 'Lin2'],
    ['batch1', 'batch2']
], names=['Lin', 'Batch'])

# 定义行索引
index = ['A', 'B']

# 创建 DataFrame
df = pd.DataFrame(data, index=index, columns=columns)

# 使用 loc 方法修改 A 配置下 Lin1 的 batch1 对应的值
df.loc['A', ('Lin1', 'batch1')] = 150.0

print(df)
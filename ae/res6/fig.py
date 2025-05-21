import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import os
import sys

# 设置工作目录
sys.path.append("/root/paper/LLMfusion")
os.chdir("/root/paper/LLMfusion")

# 配置参数
conf = "LLMSGHD"
csv_file_path = "../ae/res6.xlsx"
sheet_name = "latency"

# 从 Excel 文件中读取数据
df = pd.read_excel(csv_file_path, sheet_name=sheet_name, index_col=0, header=[0, 1])

# 提取指定行并重塑为二维数组
data = df.loc[conf]  # 提取指定行
data = data.unstack().values  # 转换为二维数组

# 生成坐标标签
input_length = [128, 512, 1024, 2048]
output_length = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]

# 创建图形和子图
fig, ax = plt.subplots(figsize=(8, 6))  # 调整图形大小

# 绘制热图，确保每个格子是正方形
from matplotlib.colors import Normalize
norm = Normalize(vmin=np.min(data), vmax=np.max(data))  # 归一化数据范围
im = ax.imshow(data, cmap='cividis', norm=norm, aspect='equal', origin='lower')

# 添加网格线
ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", size=0)  # 隐藏小刻度

# 使用 make_axes_locatable 调整颜色条的位置和大小
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 颜色条宽度为热图的5%，间距为0.1

# 添加颜色条
cbar = plt.colorbar(im, cax=cax)

# 设置 x 轴和 y 轴标签
ax.set_xticks(np.arange(len(output_length)))
ax.set_yticks(np.arange(len(input_length)))
ax.set_xticklabels(output_length)
ax.set_yticklabels(input_length)

# 添加轴标签
ax.set_xlabel('Output Length')
ax.set_ylabel('Input Length')

# 在热图中添加数值标注
for i in range(data.shape[0]):  # 遍历行
    for j in range(data.shape[1]):  # 遍历列
        value = int(np.ceil(data[i, j]))  # 对数值上取整
        ax.text(j, i, str(value), ha='center', va='center', color='white', fontsize=10, fontweight='bold')

# 保存热图
plt.savefig('./ae/res6/heatmap1.pdf', dpi=300, bbox_inches='tight')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
import os
import copy
import pandas as pd
# # 获取当前工作目录
# if __name__ == "__main__":
#     current_dir = os.getcwd()
#     print("当前目录:", current_dir)

#     # 更改到上两级目录
#     parent_dir = os.path.dirname(os.path.dirname(current_dir))
#     os.chdir(parent_dir)
#     sys.path.append(parent_dir)

#     # 验证当前工作目录
#     new_dir = os.getcwd()
#     print("更改后的目录:", new_dir)
# 假设这是你的数据矩阵，这里随机生成一个类似形状的数据，你需要替换为真实数据

sys.path.append("/root/paper/LLMfusion")
os.chdir("/root/paper/LLMfusion")
conf = "LLMSGHD"
csv_file_path = "../ae/res6.xlsx"
sheet_name = "latency"

df = pd.read_excel(csv_file_path, sheet_name=sheet_name, index_col=0, header=[0,1])

data = df.loc[conf]  # 提取指定行

# 将一维数据按二级列索引重塑为二维数组
my_data = data.unstack().values  # 转换为二维数组

data = df.loc["GA100"]
A100_data = data.unstack().values  # 转换为二维数组

data = my_data / A100_data

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
        value = data[i, j]  # 对数值上取整
        ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')

# 保存热图
plt.savefig('./ae/res6/heatmap.pdf', dpi=300, bbox_inches='tight')
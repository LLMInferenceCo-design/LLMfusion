import sys
import os
import copy
import pandas as pd
# 获取当前工作目录
if __name__ == "__main__":
    current_dir = os.getcwd()
    print("当前目录:", current_dir)

    # 更改到上两级目录
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    os.chdir(parent_dir)
    sys.path.append(parent_dir)

    # 验证当前工作目录
    new_dir = os.getcwd()
    print("更改后的目录:", new_dir)
# sys.path.append("/root/paper/LLMfusion")
# os.chdir("/root/paper/LLMfusion")
import logging
from ae.fig1.change_size import change_hardware_params
from software_model.matmul_horizontal_fusion import HorizontalMatmulFusion
from software_model.mutmul_fusion import MatmulFusion
from software_model.matmul import Matmul, BatchedMatmul
from software_model.DataFrame import DataType, Tensor, data_type_dict
from hardware_model.system import System
from software_model.softmax import Softmax
from software_model.flash_attention_fusion import FlashAttentionFusion
from LLM_model.opt175b import opt175b_prefill, opt175b_decode
import json
import time
from loguru import logger
from util.mapping import Mapping
from multiprocessing import Pool, cpu_count
from openpyxl import load_workbook


csv_file_path = "../ae/res5-d.xlsx"
sheet_name = "decode"
output_file_path = "../ae/res5_decode.xlsx"  # 输出文件路径

# 读取 Excel 文件
read_df = pd.read_excel(csv_file_path, header=[0, 1, 2], index_col=0, sheet_name=sheet_name)

# 遍历每个值并加 1
modified_df = read_df.copy()  # 创建一个副本以存储修改后的数据
for index, row in read_df.iterrows():
    for col in read_df.columns:
        modified_df.at[index, col] = row[col] / read_df.loc["total_latency", (col[0], col[1], "GA100")] # 每个值加 1

# 将修改后的 DataFrame 保存到新的 Excel 文件
with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    modified_df.to_excel(writer, sheet_name=sheet_name)

print(f"修改后的数据已保存到 {output_file_path}")

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # 读取 Excel 文件
# csv_file_path = "../ae/res5.xlsx"
# sheet_name = "prefill"
# df = pd.read_excel(csv_file_path, header=[0, 1, 2], index_col=0, sheet_name=sheet_name)

# # 提取 Batch Size 相关数据作为 x 轴标签
# batch_sizes = df['Batch Size'].dropna().astype(int)

# # 提取 Config 相关数据作为不同的堆积类别
# configs = df.columns[2::4]

# # 初始化图形
# fig, ax = plt.subplots(figsize=(12, 6))

# # 计算每个类别在 x 轴上的位置
# x = range(len(batch_sizes))
# width = 0.2

# # 循环绘制堆积柱状图
# bottom = np.zeros(len(batch_sizes))
# for i, config in enumerate(configs):
#     data = df[config].dropna()
#     ax.bar([j + i * width for j in x], data, width, bottom=bottom, label=config)
#     bottom += data

# # 设置 x 轴标签
# ax.set_xticks([j + (len(configs) - 1) * width / 2 for j in x])
# ax.set_xticklabels(batch_sizes)

# # 设置图表标题和坐标轴标签
# ax.set_title('图表标题')
# ax.set_xlabel('Batch Size')
# ax.set_ylabel('Latency')

# # 显示图例
# ax.legend()
# plt.savefig("wide_chart.jpg", dpi=300)
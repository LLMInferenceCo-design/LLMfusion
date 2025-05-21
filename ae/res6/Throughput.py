import sys
import os
import copy
import pandas as pd
# 获取当前工作目录
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
sys.path.append("/root/paper/LLMfusion")
os.chdir("/root/paper/LLMfusion")
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
config = ["LLMCompass_Latency", "LLMCompass_Throught", "GA100", "me_prefill", "me_decode"]
core_count = [64, 64, 128, 32, 128]
sublane_count = [4, 4, 4, 4, 4]
vector_width = [32, 32, 32, 32, 16]
array_height = [16, 32, 16, 64, 16]
array_width = [16, 32, 16, 64, 16]
SRAM_KB = [192, 768, 192, 1024, 128]
global_buffer_MBS = [24, 48, 48, 48, 24]

global_buffer_bandwidth_per_cycle_bytes = [2560, 5120, 5120, 5120, 7680]
memory_bandwidths = [5, 2.5, 5, 2.5, 5]
total_capacity_GBS = [80, 512, 80, 80, 80]
memory_protocols = ["HBM2e", "PCIe5", "HBM2e", "PCIe5", "HBM2e"]
device_count = 4

batch_size = 8

if __name__ == "__main__":
    Lin = [128, 512, 1024, 2048]
    Lout = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
    # Lout = [i-1 for i in Lout]
    L = [128*i for i in range(1, 34)]
    csv_file_path = "../ae/res6.xlsx"
    index = config.copy()[:3]
    index.append("LLMSGHD")
    cols = []
    cols.append(Lin)
    cols.append(Lout)
    columns = pd.MultiIndex.from_product(cols, names=["Lin", "Lout"])

    df = pd.DataFrame(index=index, columns=columns)

    df_prefill = pd.read_excel(csv_file_path, sheet_name="prefill", index_col=0, header=0)
    df_decode = pd.read_excel(csv_file_path, sheet_name="decode", index_col=0, header=0)
    for cof in config[:3]:
        for lin in Lin:
            for lout in Lout:
                latency_prefill = df_prefill.loc[cof, lin] * 96
                latency_decode = 0
                num = 0
                idx = L.index(lin)
                l=L[idx]
                lall = lin+ lout
                while l <= lall:
                    latency_decode += df_decode.loc[cof, l-1] * 96
                    num += 1
                    idx += 1
                    l = L[idx]
                    
                df.loc[cof, (lin, lout)] = batch_size * lout / (latency_prefill + (latency_decode / num)*lout)

    conf = "LLMSGHD"
    for lin in Lin:
        for lout in Lout:
            latency_prefill = df_prefill.loc["me_prefill", lin] * 96
            latency_decode = 0
            num = 0
            idx = L.index(lin)
            l=L[idx]
            lall = lin+ lout
            while l <= lall:
                latency_decode += df_decode.loc["me_decode", l-1] * 96
                num += 1
                idx += 1
                l = L[idx]
                
            df.loc[conf, (lin, lout)] = batch_size * lout /(latency_prefill + (latency_decode / num)*lout)

    # Ensure the sheet "latency" does not already exist before appending
    with pd.ExcelWriter(csv_file_path, engine='openpyxl', mode='a') as writer:
        workbook = writer.book
        if "latency" in workbook.sheetnames:
            del workbook["latency"]
        df.to_excel(writer, sheet_name="latency", index=True)

                


    


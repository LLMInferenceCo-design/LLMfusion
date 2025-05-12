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
from LLM_model.opt175b import opt175b_prefill,opt175b_decode
import json
import time
from loguru import logger
from util.mapping import Mapping
from multiprocessing import Pool, cpu_count
from openpyxl import load_workbook

config = ['A', 'B', 'C', 'D', 'E']
core_count = [8, 32, 128, 256, 512]
sublane_count = 4
vector_width = [512, 128, 32, 16, 8]
array_height = [64, 32, 16, 8, 4]
array_width = [64, 32, 16, 16, 16]
SRAM_KB = [3072, 768, 192, 96, 48]

SRAM = [64, 128, 192,256, 512, 1024]
batch_size = 8
Lin = 2048
Lout = 1023
device_count = 4

def change_config_prefill_test(hardware_config):
    M = Lin
    prefill = opt175b_prefill(12288, 96, device_count, data_type=data_type_dict['fp16'])
    _ = prefill(Tensor([batch_size, M, 12288]))
    with open('./configs/GA100.json', "r") as f:
        arch_specs = json.load(f)
    system, area = change_hardware_params(hardware_config, arch_specs)
    latency = prefill.compile_and_simulate(system)

    return latency,area

def change_config_decode_test(hardware_config):
    M = 1
    decode = opt175b_decode(12288, 96, device_count, data_type=data_type_dict['fp16'])
    _ = decode(Tensor([batch_size, M, 12288]), Lout+Lin)
    with open('./configs/GA100.json', "r") as f:
        arch_specs = json.load(f)
    system, area = change_hardware_params(hardware_config, arch_specs)
    latency = decode.compile_and_simulate(system)

    return latency,area

def process_config(args):
    hardware_config, i, j, test_function = args
    hardware_config['config'] = config[j]
    hardware_config['core_count'] = core_count[j]
    hardware_config['sublane_count'] = sublane_count
    hardware_config['vector_width'] = vector_width[j]
    hardware_config['array_height'] = array_height[j]
    hardware_config['array_width'] = array_width[j]
    # hardware_config['SRAM_KB'] = SRAM_KB[j]
    hardware_config["SRAM_KB"] = SRAM[i]
    latency, area = test_function(hardware_config)
    print(f"Config {config[j]}: mem:{SRAM[i]}  : Latency: {latency}, Area: {area}")
    return i,area,latency

def save_to_xlsx(csv_file_path, sheet_name, results):
    """将数据保存到Excel文件"""
    filtered_data = [(SRAM[index], values) for index, _, values in results]
    # 创建一个空的 DataFrame
    df = pd.DataFrame()

    # 遍历过滤后的数据
    for index, values in filtered_data:
        # 把每个子字典转换为 DataFrame 的一行，并设置索引
        row = pd.DataFrame(values, index=[index])
        # 将行合并到主 DataFrame 中
        df = pd.concat([df, row], axis=0)

    # 检查文件是否存在
    if os.path.exists(csv_file_path):
        # 读取 Excel 文件
        excel_file = pd.ExcelFile(csv_file_path)
        # 获取所有表名
        sheet_names = excel_file.sheet_names
        with pd.ExcelWriter(csv_file_path, mode='a', if_sheet_exists='replace') as writer:
            # 如果工作表存在，先删除
            if sheet_name in sheet_names:
                df.to_excel(writer, sheet_name=sheet_name)
            else:
                df.to_excel(writer, sheet_name=sheet_name)
    else:
        # 文件不存在，直接写入
        df.to_excel(csv_file_path, sheet_name=sheet_name)

if __name__ == "__main__":
    hardware_config = {
        "config": "A100",
        'core_count': 128,
        "sublane_count": 4,
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'SRAM_KB': 192,
        "memory_bandwidth": 1,
    }

    start_time = time.time()
    # logger.info(f"Start time: {start_time}")
    loc = config.index('C')

    task =[]
    for i in range(1, len(SRAM)):
        task.append((copy.deepcopy(hardware_config), i, loc, change_config_prefill_test))
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_config, task)
    csv_file_path = "../ae/res4.xlsx"
    sheet_name = "prefill"
    save_to_xlsx(csv_file_path, sheet_name, results)

    loc = config.index('C')
    task =[]
    for i in range(len(SRAM)):
        task.append((copy.deepcopy(hardware_config), i, loc, change_config_decode_test))
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_config, task)
    csv_file_path = "../ae/res4.xlsx"
    sheet_name = "decode"
    save_to_xlsx(csv_file_path, sheet_name, results)
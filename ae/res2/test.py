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

config = ['A', 'B', 'C', 'D', 'E']
core_count = [8, 32, 128, 256, 512]
sublane_count = 4
vector_width = [512, 128, 32, 16, 8]
array_height = [64, 32, 16, 8, 4]
array_width = [64, 32, 16, 16, 16]
SRAM_KB = [3072, 768, 192, 96, 48]

batch_size = 8
Lin = 2048
Lout = 1023
device_count = 16

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

if __name__ == "__main__":
    hardware_config = {
        "config": "A100",
        'core_count': 128,
        "sublane_count": 4,
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'SRAM_KB': 192,
    }

    # latency, area = change_config_prefill_test(hardware_config)
    # print(f"Prefill latency: {latency}, area: {area}")
    latency,area = change_config_decode_test(hardware_config)
    print(f"Decode latency: {latency}, area: {area}")


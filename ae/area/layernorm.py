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
from software_model.layernorm import LayerNorm
from LLM_model.opt175b import opt175b_prefill,opt175b_decode
import json
import time
from loguru import logger
from util.mapping import Mapping
from multiprocessing import Pool, cpu_count
from openpyxl import load_workbook

model = LayerNorm(data_type=data_type_dict['fp16'])
b=8
m =1
n = 12288

_ = model(Tensor([b, m, n]))

hardware_config = {
        "config": "A100",
        'core_count': 128,
        "sublane_count": 4,
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'SRAM_KB': 192,
    }
hardware_config = {
        "config": "A100",
        'core_count':512,
        "sublane_count": 4,
        "array_width": 16,
        "array_height": 4,
        'vector_width': 8,
        'SRAM_KB': 256,
    }

with open('./configs/GA100.json', "r") as f:
    arch_specs = json.load(f)
system, area = change_hardware_params(hardware_config, arch_specs)

times = model.compile_and_simulate(system.device) * 2
print("times:", times)
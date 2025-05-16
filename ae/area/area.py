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
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
import json
from ae.fig1.change_size import change_hardware_params

config = ["LLMCompass_Latency", "LLMCompass_Throught", "GA100", "me_prefill"]
core_count = [64, 64, 128, 32]
sublane_count = [4, 4, 4,4]
vector_width = [32, 32, 32, 32]
array_height = [16, 32, 16, 64]
array_width = [16, 32, 16, 64]
SRAM_KB = [192, 768, 192, 1024]
global_buffer_MBS = [24, 48, 48, 48]

global_buffer_bandwidth_per_cycle_bytes = [2560, 5120, 5120, 5120]
memory_bandwidths = [5, 2.5, 5, 2.5]
total_capacity_GBS = [80, 512, 80, 80]
memory_protocols = ["HBM2e", "PCIe5", "HBM2e", "PCIe5"]
device_count = 4

def process_config(args):
    """单独处理一个硬件配置的函数，用于多进程"""
    hardware_config,  i = args
    hardware_config["config"] = config[i]
    hardware_config["core_count"] = core_count[i]
    hardware_config["sublane_count"] = sublane_count[i]
    hardware_config["vector_width"] = vector_width[i]
    hardware_config["array_width"] = array_width[i]
    hardware_config["array_height"] = array_height[i]
    hardware_config["SRAM_KB"] = SRAM_KB[i]
    hardware_config["global_buffer_MB"] = global_buffer_MBS[i]
    hardware_config["global_buffer_bandwidth_per_cycle_byte"] = global_buffer_bandwidth_per_cycle_bytes[i]
    hardware_config["memory_bandwidth"] = memory_bandwidths[i]
    hardware_config["total_capacity_GB"] = total_capacity_GBS[i]
    hardware_config["memory_protocol"] = memory_protocols[i]

    with open("./configs/GA100.json", "r") as f:
        configs_dict = json.load(f)
    system, area = change_hardware_params(hardware_config, configs_dict)
    print (f"Config: {config[i]}, area:{area} pass")

for i in range(len(config)):
    hardware_config = {
        "config": "A100",
        'core_count': 128,
        "sublane_count": 4,
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'SRAM_KB': 192,
        'global_buffer_MB': 48,
        'global_buffer_bandwidth_per_cycle_byte': 5120,
        'memory_bandwidth': 5,
        'total_capacity_GB': 80,
        'memory_protocol': "HBM2",
    }
    process_config((hardware_config,i))
import sys
import os
sys.path.append("/root/paper/LLMfusion")
os.chdir("/root/paper/LLMfusion")
import logging
from ae.fig1.change_size import chage_hardware_params
from software_model.matmul_horizontal_fusion import HorizontalMatmulFusion
from software_model.mutmul_fusion import MatmulFusion
from software_model.matmul import Matmul, BatchedMatmul
from software_model.DataFrame import DataType, Tensor, data_type_dict
from hardware_model.system import System
from software_model.softmax import Softmax
from software_model.flash_attention_fusion import FlashAttentionFusion
from LLM_model.opt175b import opt175b_prefill
import json
import time
from loguru import logger
from util.mapping import Mapping
from multiprocessing import Pool, cpu_count

config = ['A', 'B', 'C', 'D', 'E']
core_count = [8, 32, 128, 256, 512]
sublane_count = 4
vector_width = [512, 128, 32, 16, 8]
array_height = [64, 32, 16, 8, 4]
array_width = [64, 32, 16, 16, 16]
SRAM_KB = [3072, 768, 192, 96, 48]

def change_config_prefill_test(config, batch_size:int, Lin:int, device_count:int):
    Nhead = 96
    # batch_size = 1
    M = Lin
    N = 128
    kv_cache = 0

    assert Nhead % device_count == 0
    batch = batch_size * Nhead//device_count
    with open('./configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system, area = chage_hardware_params(config, arch_specs)

    QK = BatchedMatmul(data_type=data_type_dict['fp16'])
    _ = QK(Tensor([batch, M, N]), Tensor([batch, N, M + kv_cache]))

    S = Softmax(data_type=data_type_dict['fp16'])
    _ = S(Tensor([batch * M, M + kv_cache]))

    SV = BatchedMatmul(data_type=data_type_dict['fp16'])
    _ = SV(Tensor([batch, M, M + kv_cache]), Tensor([batch, M + kv_cache, N]))

    flash_attention = FlashAttentionFusion([QK, S, SV], data_type_dict['fp16'])
    time_s = flash_attention.compile_and_simulate(system.device)
    # tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)
    clock = time_s * system.device.compute_module.clock_freq
    # logger.debug(f"Prefill config: {config['config']}, batch_size: {batch_size}, Lin: {Lin}, device_count: {device_count}, clock num: {clock}")
    return time_s, area, clock

def change_config_decode_test(config, batch_size:int, Lin:int, device_count:int):
    Nhead = 96
    # batch_size = 1
    M = 1
    N = 128
    kv_cache = Lin

    assert Nhead % device_count == 0
    batch = batch_size * Nhead//device_count
    with open('./configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system, area = chage_hardware_params(config, arch_specs)

    QK = BatchedMatmul(data_type=data_type_dict['fp16'])
    _ = QK(Tensor([batch, M, N]), Tensor([batch, N, M + kv_cache]))

    S = Softmax(data_type=data_type_dict['fp16'])
    _ = S(Tensor([batch * M, M + kv_cache]))

    SV = BatchedMatmul(data_type=data_type_dict['fp16'])
    _ = SV(Tensor([batch, M, M + kv_cache]), Tensor([batch, M + kv_cache, N]))

    flash_attention = FlashAttentionFusion([QK, S, SV], data_type_dict['fp16'])
    time_s = flash_attention.compile_and_simulate(system.device)
    # tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)
    clock = time_s * system.device.compute_module.clock_freq
    # logger.debug(f"Prefill config: {config['config']}, batch_size: {batch_size}, Lin: {Lin}, device_count: {device_count}, clock num: {clock}")
    return time_s, area, clock

def process_config(args):
    """单独处理一个硬件配置的函数，用于多进程"""
    hardware_config, b, l, device_count, i, test_function = args
    hardware_config['config'] = config[i]
    hardware_config['core_count'] = core_count[i]
    hardware_config['sublane_count'] = sublane_count
    hardware_config['vector_width'] = vector_width[i]
    hardware_config['array_height'] = array_height[i]
    hardware_config['array_width'] = array_width[i]
    hardware_config['SRAM_KB'] = SRAM_KB[i]
    time_s, area, clock = test_function(hardware_config, b, l, device_count)
    return config[i], b, l, time_s, area, clock  # 返回更多信息

if __name__ == '__main__':
    hardware_config = {
        "config": "A100",
        'core_count': 128,
        "sublane_count": 4,
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'SRAM_KB': 192,
    }
    batch_size = [1, 4, 16, 64, 256]
    Lin = [128, 1024, 2048]
    Lout = [128, 1024, 2048, 4096]
    device_count = 16

    time_s, area, clock = change_config_prefill_test(hardware_config, 1, 128, device_count)
    print("time_s:", time_s, "clock:",clock )
    ## 构造任务列表
    # tasks = []
    # for l in Lin:
    #     for b in batch_size:
    #         for i in range(4):
    #             tasks.append((hardware_config.copy(), b, l, device_count, i, change_config_prefill_test))

    # # 使用多进程池并行处理
    # with Pool(processes=cpu_count()) as pool:
    #     results = pool.map(process_config, tasks)

    # # 处理结果
    # for result in results:
    #     config_name, batch_size, Lin, time_s, area, clock = result
    #     print(f"Prefill Config: {config_name}, Batch Size: {batch_size}, Lin: {Lin}, Time: {time_s}, area: {area},Clock: {clock}")
    
    ##  构造任务列表
    tasks = []
    for l in Lout:
        for b in batch_size:
            for i in range(5):
                tasks.append((hardware_config.copy(), b, l, device_count, i, change_config_decode_test))

    # 使用多进程池并行处理
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_config, tasks)

    # 处理结果
    for result in results:
        config_name, batch_size, Lout, time_s, area, clock = result
        print(f"Decode Config: {config_name}, Batch Size: {batch_size}, Lin: {Lout}, Time: {time_s}, area: {area},Clock: {clock}")
    
    # start_time = time.time()
    # batch_size =8
    # Nhead = 96
    # # batch_size = 1
    # M = 2048
    # N = 128
    # kv_cache = 0
    

    # assert Nhead % device_count == 0
    # batch = batch_size * Nhead//device_count


    # with open('./configs/ga102_template.json', "r") as f:
    #     arch_specs = json.load(f)
    # system = chage_hardware_params(hardware_config, arch_specs)

    # QK = BatchedMatmul(data_type=data_type_dict['fp16'])
    # _ = QK(Tensor([batch, M, N]), Tensor([batch, N, M + kv_cache]))

    # S = Softmax(data_type=data_type_dict['fp16'])
    # _ = S(Tensor([batch * M, M + kv_cache]))

    # SV = BatchedMatmul(data_type=data_type_dict['fp16'])
    # _ = SV(Tensor([batch, M, M + kv_cache]), Tensor([batch, M + kv_cache, N]))

    # flash_attention = FlashAttentionFusion([QK, S, SV], data_type_dict['fp16'])

    # time_s = flash_attention.compile_and_simulate(system.device)
    # # tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)
    # clock = time_s * system.device.compute_module.clock_freq

    # end_time = time.time()
    # logger.info(f"OperatorFusion Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (end_time - start_time))

    

    # mapping_display(flash_attention.best_mapping)
    # tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)

    # logger.info(f"Execution time: %s seconds\ntimes: {tmp}")
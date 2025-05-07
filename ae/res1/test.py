import sys
import os
import copy
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
from LLM_model.opt175b import opt175b_prefill
import json
import time
from loguru import logger
from util.mapping import Mapping
from multiprocessing import Pool, cpu_count
import copy

if __name__ == '__main__':
    # hardware_config = {
    #     "config": "E",
    #     'core_count': 512,
    #     "sublane_count": 4,
    #     "array_width": 16,
    #     "array_height": 4,
    #     'vector_width': 8,
    #     'SRAM_KB': 48,
    # }
    hardware_config = {
        "config": "A100",
        'core_count': 128,
        "sublane_count": 4,
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'SRAM_KB':192,
    }

    start_time = time.time()
    batch_size =1
    Nhead = 96
    # batch_size = 1
    M = 1
    N = 128
    kv_cache = 2048
    device_count = 16

    assert Nhead % device_count == 0
    batch = batch_size * Nhead//device_count


    # with open('./configs/GA100.json', "r") as f:
    #     arch_specs = json.load(f)
    # system, _ = change_hardware_params(hardware_config, arch_specs)
    #
    # QK = BatchedMatmul(data_type=data_type_dict['fp16'])
    # _ = QK(Tensor([batch, M, N]), Tensor([batch, N, M + kv_cache]))
    #
    # S = Softmax(data_type=data_type_dict['fp16'])
    # _ = S(Tensor([batch * M, M + kv_cache]))
    #
    # SV = BatchedMatmul(data_type=data_type_dict['fp16'])
    # _ = SV(Tensor([batch, M, M + kv_cache]), Tensor([batch, M + kv_cache, N]))
    #
    # flash_attention = FlashAttentionFusion([QK, S, SV], data_type_dict['fp16'])
    #
    # time_s = flash_attention.compile_and_simulate(system.device)
    # # tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)
    # clock = time_s * system.device.compute_module.clock_freq
    #
    # end_time = time.time()
    # logger.info(f"OperatorFusion Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (end_time - start_time))
    #
    #
    #
    # tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)
    #
    # logger.info(f"Execution time: %s seconds\ntimes: {tmp}")

    with open('./configs/GA100.json', "r") as f:
        arch_specs = json.load(f)
    system, area = change_hardware_params(hardware_config, arch_specs)

    mul1 = Matmul(data_type=data_type_dict['fp16'])

    _ = mul1(Tensor([M, N]), Tensor([N, M + kv_cache]))

    mul1_fusion = MatmulFusion([mul1], data_type_dict['fp16'])

    mul_fusion = [copy.deepcopy(mul1_fusion) for _ in range(batch)]

    mul1 = HorizontalMatmulFusion(mul_fusion, data_type_dict['fp16'])
    QK_times = mul1.compile_and_simulate(system.device)
    QK_clock = QK_times * system.device.compute_module.clock_freq
    print("QK_times", QK_times, "QK_clock", QK_clock)

    S = Softmax(data_type=data_type_dict['fp16'])
    _ = S(Tensor([batch * M, M + kv_cache]))
    S_times = S.compile_and_simulate(system.device)
    S_clock = S_times * system.device.compute_module.clock_freq
    print("S_times", S_times, "S_clock", S_clock)

    mul2 = Matmul(data_type=data_type_dict['fp16'])
    _ = mul2(Tensor([M, M + kv_cache]), Tensor([M + kv_cache, N]))
    mul2_fusion = MatmulFusion([mul2], data_type_dict['fp16'])
    mul_fusion = [copy.deepcopy(mul2_fusion) for _ in range(batch)]
    mul2 = HorizontalMatmulFusion(mul_fusion, data_type_dict['fp16'])
    SV_times = mul2.compile_and_simulate(system.device)
    SV_clock = SV_times * system.device.compute_module.clock_freq
    print("SV_times", SV_times, "SV_clock", SV_clock)

    time_s = QK_times + S_times + SV_times
    clock = QK_clock + S_clock + SV_clock
    logger.info(f"Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (time_s))
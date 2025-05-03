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

import logging
from ae.fig1.change_size import change_hardware_params
from software_model.matmul_horizontal_fusion import HorizontalMatmulFusion
from software_model.mutmul_fusion import MatmulFusion
from software_model.matmul import Matmul, BatchedMatmul
from software_model.DataFrame import DataType, Tensor, data_type_dict
from hardware_model.system import System
import json
import time
from loguru import logger
from util.mapping import Mapping

def mapping_display(M:Mapping):
    print(f"l2_tile_M = {M.l2_tile_M}, ")
    print(f"l2_tile_N = {M.l2_tile_N}, ")
    print(f"l2_tile_K = {M.l2_tile_K}, ")
    print("is_l2_double_buffering = True,")
    print(f"l1_tile_M = {M.l1_tile_M}, ")
    print(f"l1_tile_N = {M.l1_tile_N}, ")
    print(f"l1_tile_K = {M.l1_tile_K}, ")
    print("l2_loop_order = 'knm',")
    print("l1_loop_order = 'knm',")
    print("l0_M_tiling_factor =", M.l0_M_tiling_factor,",")
    print("l0_N_tiling_factor =", M.l0_N_tiling_factor,",")
    print("l0_K_tiling_factor =", M.l0_K_tiling_factor,",")

if __name__ == '__main__':
    hardware_config = {
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'core_count': 128,
        'SRAM_KB': 96,

    }
    start_time = time.time()
    M = 1
    K = 128
    N= 4096
    # logger.info(f"Start time: {start_time}")
    with open('./configs/GA100.json', "r") as f:
        arch_specs = json.load(f)
    system,_ = change_hardware_params(hardware_config, arch_specs)

    mul1 = Matmul(data_type= data_type_dict['fp16'])
    mul2 = Matmul(data_type= data_type_dict['fp16'])
    mul3 = Matmul(data_type= data_type_dict['fp16'])

    _ = mul1(Tensor([M, K]), Tensor([K, N]))
    _ = mul2(Tensor([M, K]), Tensor([K, N]))
    _ = mul3(Tensor([M, K]), Tensor([K, N]))

    mul1_fusion = MatmulFusion([mul1], data_type_dict['fp16'])
    mul2_fusion = MatmulFusion([mul2], data_type_dict['fp16'])
    mul3_fusion = MatmulFusion([mul3], data_type_dict['fp16'])
    mul1 = HorizontalMatmulFusion([mul1_fusion], data_type_dict['fp16'])
    time_s = mul1.compile_and_simulate(system.device)
    clock = time_s * system.device.compute_module.clock_freq
    logger.info(f" times: {time_s}     clock num: {clock}")
    # mapping_display(mul1.best_mapping)
    clock = mul1.simulate(mul1.best_mapping, system.device)

    mul_hor_fusion = HorizontalMatmulFusion([mul1_fusion, mul2_fusion, mul3_fusion], data_type_dict['fp16'])

    time_s = mul_hor_fusion.compile_and_simulate(system.device)
    clock = time_s * system.device.compute_module.clock_freq

    end_time = time.time()

    logger.info(f"Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (end_time - start_time))
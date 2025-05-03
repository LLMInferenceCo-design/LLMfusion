import sys
import os
sys.path.append("/root/paper/LLMfusion")
os.chdir("/root/paper/LLMfusion")
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

if __name__ == '__main__':
    hardware_config = {
        "array_width": 16,
        "array_height": 8,
        'vector_width': 16,
        'core_count': 216,
        'SRAM_KB': 96,

    }
    start_time = time.time()
    M = 2048
    K = 12288
    N= 3072
    # logger.info(f"Start time: {start_time}")
    with open('./configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system = change_hardware_params(hardware_config, arch_specs)

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
    clock = mul1.simulate(mul1.best_mapping, system.device)

    mul_hor_fusion = HorizontalMatmulFusion([mul1_fusion, mul2_fusion, mul3_fusion], data_type_dict['fp16'])

    time_s = mul_hor_fusion.compile_and_simulate(system.device)
    clock = time_s * system.device.compute_module.clock_freq

    end_time = time.time()

    logger.info(f"Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (end_time - start_time))
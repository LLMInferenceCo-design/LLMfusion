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



if __name__ == '__main__':
    hardware_config = {
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'core_count': 128,
        'SRAM_KB': 96,

    }

    start_time = time.time()
    batch_size =8
    Nhead = 96
    # batch_size = 1
    M = 2048
    N = 128
    kv_cache = 0
    device_count = 4

    assert Nhead % device_count == 0
    batch = batch_size * Nhead//device_count


    with open('./configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system = chage_hardware_params(hardware_config, arch_specs)

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

    end_time = time.time()
    logger.info(f"OperatorFusion Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (end_time - start_time))

    

    # mapping_display(flash_attention.best_mapping)
    # tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)

    # logger.info(f"Execution time: %s seconds\ntimes: {tmp}")
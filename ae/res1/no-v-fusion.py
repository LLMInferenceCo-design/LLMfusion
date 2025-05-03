import logging
from ae.fig1.change_size import change_hardware_params
from software_model.matmul_horizontal_fusion import HorizontalMatmulFusion
from software_model.mutmul_fusion import MatmulFusion
from software_model.matmul import Matmul, BatchedMatmul
from software_model.DataFrame import DataType, Tensor, data_type_dict
from software_model.softmax import Softmax
from hardware_model.system import System
import json
import time
from loguru import logger
from util.mapping import Mapping
import copy

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

def change_config_prefill_no_V_test(config, batch_size:int, Lin:int, device_count:int):
    Nhead = 96
    # batch_size = 1
    M = Lin
    N = 128
    kv_cache = 0

    assert Nhead % device_count == 0
    batch = batch_size * Nhead//device_count
    with open('./configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system, area = change_hardware_params(config, arch_specs)


    mul1 = Matmul(data_type=data_type_dict['fp16'])

    _ = mul1(Tensor([M, N]), Tensor([N, M+kv_cache]))

    mul1_fusion = MatmulFusion([mul1], data_type_dict['fp16'])

    mul_fusion = [copy.deepcopy(mul1_fusion) for _ in range(batch)]

    mul1 = HorizontalMatmulFusion(mul_fusion, data_type_dict['fp16'])
    QK_times = mul1.compile_and_simulate(system.device)
    QK_clock = QK_times * system.device.compute_module.clock_freq

    S = Softmax(data_type=data_type_dict['fp16'])
    _ = S(Tensor([batch * M, M + kv_cache]))
    S_times = S.compile_and_simulate(system.device)
    S_clock = S_times * system.device.compute_module.clock_freq

    mul2 = Matmul(data_type=data_type_dict['fp16'])
    _ = mul2(Tensor([M, M + kv_cache]), Tensor([M + kv_cache, N]))
    mul2_fusion = MatmulFusion([mul2], data_type_dict['fp16'])
    mul_fusion = [copy.deepcopy(mul2_fusion) for _ in range(batch)]
    mul2 = HorizontalMatmulFusion(mul_fusion, data_type_dict['fp16'])
    SV_times = mul2.compile_and_simulate(system.device)
    SV_clock = SV_times * system.device.compute_module.clock_freq

    time_s = QK_times + S_times + SV_times
    clock = QK_clock + S_clock + SV_clock

    return time_s, area, clock


def change_config_decode_no_V_test(config, batch_size: int, Lin: int, device_count: int):
    Nhead = 96
    # batch_size = 1
    M = 1
    N = 128
    kv_cache = Lin

    assert Nhead % device_count == 0
    batch = batch_size * Nhead // device_count
    with open('./configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system, area = change_hardware_params(config, arch_specs)

    mul1 = Matmul(data_type=data_type_dict['fp16'])

    _ = mul1(Tensor([M, N]), Tensor([N, M + kv_cache]))

    mul1_fusion = MatmulFusion([mul1], data_type_dict['fp16'])

    mul_fusion = [copy.deepcopy(mul1_fusion) for _ in range(batch)]

    mul1 = HorizontalMatmulFusion(mul_fusion, data_type_dict['fp16'])
    QK_times = mul1.compile_and_simulate(system.device)
    QK_clock = QK_times * system.device.compute_module.clock_freq

    S = Softmax(data_type=data_type_dict['fp16'])
    _ = S(Tensor([batch * M, M + kv_cache]))
    S_times = S.compile_and_simulate(system.device)
    S_clock = S_times * system.device.compute_module.clock_freq

    mul2 = Matmul(data_type=data_type_dict['fp16'])
    _ = mul2(Tensor([M, M + kv_cache]), Tensor([M + kv_cache, N]))
    mul2_fusion = MatmulFusion([mul2], data_type_dict['fp16'])
    mul_fusion = [copy.deepcopy(mul2_fusion) for _ in range(batch)]
    mul2 = HorizontalMatmulFusion(mul_fusion, data_type_dict['fp16'])
    SV_times = mul2.compile_and_simulate(system.device)
    SV_clock = SV_times * system.device.compute_module.clock_freq

    time_s = QK_times + S_times + SV_times
    clock = QK_clock + S_clock + SV_clock

    return time_s, area, clock

def process_config(args):
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
        "array_width": 16,
        "array_height": 16,
        'vector_width': 32,
        'core_count': 128,
        'SRAM_KB': 96,

    }
    start_time = time.time()
    batch = 256
    M = 1
    K = 128
    N= 4097
    # logger.info(f"Start time: {start_time}")
    with open('./configs/GA100.json', "r") as f:
        arch_specs = json.load(f)
    system = change_hardware_params(hardware_config, arch_specs)

    mul1 = Matmul(data_type= data_type_dict['fp16'])

    _ = mul1(Tensor([M, K]), Tensor([K, N]))
   

    mul1_fusion = MatmulFusion([mul1], data_type_dict['fp16'])

    mul_fusion = [copy.deepcopy(mul1_fusion) for _ in range(batch)]

    mul1 = HorizontalMatmulFusion(mul_fusion, data_type_dict['fp16'])
    time_s = mul1.compile_and_simulate(system.device)
    clock = time_s * system.device.compute_module.clock_freq
    logger.info(f" times: {time_s}     clock num: {clock}")
    # mapping_display(mul1.best_mapping)
    # clock = mul1.simulate(mul1.best_mapping, system.device)

    # mul_hor_fusion = HorizontalMatmulFusion([mul1_fusion, mul2_fusion, mul3_fusion], data_type_dict['fp16'])

    # time_s = mul_hor_fusion.compile_and_simulate(system.device)
    # clock = time_s * system.device.compute_module.clock_freq

    # end_time = time.time()

    # logger.info(f"Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (end_time - start_time))
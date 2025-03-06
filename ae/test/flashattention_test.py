from ae.fig1.change_size import chage_hardware_params
from software_model.matmul_horizontal_fusion import HorizontalMatmulFusion
from software_model.mutmul_fusion import MatmulFusion
from software_model.matmul import Matmul, BatchedMatmul
from software_model.DataFrame import DataType, Tensor, data_type_dict
from hardware_model.system import System
from software_model.softmax import Softmax
from util.mapping import Mapping
from software_model.flash_attention_fusion import FlashAttentionFusion
import json
import time
from loguru import logger

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

        }
    start_time = time.time()
    batch_size = 128 *  96
    # batch_size = 1
    M = 2048
    N = 128
    kv_cache = 0

    with open('./configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system = chage_hardware_params(hardware_config, arch_specs)

    QK = BatchedMatmul(data_type=data_type_dict['fp16'])
    _ = QK(Tensor([batch_size, M, N]), Tensor([batch_size, N, M + kv_cache]))

    S = Softmax(data_type=data_type_dict['fp16'])
    _ = S(Tensor([batch_size * M, M + kv_cache]))

    SV = BatchedMatmul(data_type=data_type_dict['fp16'])
    _ = SV(Tensor([batch_size, M, M + kv_cache]), Tensor([batch_size, M + kv_cache, N]))

    flash_attention = FlashAttentionFusion([QK, S, SV], data_type_dict['fp16'])

    time_s = flash_attention.compile_and_simulate(system.device)
    # tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)
    clock = time_s * system.device.compute_module.clock_freq

    end_time = time.time()
    logger.info(f"Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (end_time - start_time))

    # mapping_display(flash_attention.best_mapping)
    tmp = flash_attention.simulate(flash_attention.best_mapping, system.device)

    logger.info(f"Execution time: %s seconds\ntimes: {tmp}")


from ae.fig1.change_size import change_hardware_params
from software_model.matmul_horizontal_fusion import HorizontalMatmulFusion
from software_model.mutmul_fusion import MatmulFusion
from software_model.matmul import Matmul, BatchedMatmul
from software_model.gelu import GeLU
from software_model.ffn_fusion import FFNFusion
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
    print(f'l2_tile_H = {M.l2_tile_H},')
    print(f'l1_tile_H = {M.l1_tile_H},')


if __name__ == '__main__':
    hardware_config = {
        "array_width": 16,
        "array_height": 16,

    }
    with open('./configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system = change_hardware_params(hardware_config, arch_specs)

    start_time = time.time()
    device_count = 4
    batch_size = 8 * 96
    # batch_size = 1
    M = 2048 * 8
    d = 12288
    d_t = 4 *d //device_count

    H1 = Matmul(data_type= data_type_dict['fp16'])
    H_gelu = GeLU(data_type= data_type_dict['fp16'])
    H2 = Matmul(data_type= data_type_dict['fp16'])

    _ = H1(Tensor([M, d]), Tensor([d, d_t]))
    _ = H_gelu(Tensor([M, d_t]))
    _ = H2(Tensor([M, d_t]), Tensor([d_t, d]))

    ffn_fusion = FFNFusion([H1, H_gelu, H2], data_type_dict['fp16'])
    time_s = ffn_fusion.compile_and_simulate(system.device)
    clock = time_s * system.device.compute_module.clock_freq
    end_time = time.time()
    logger.info(f"Execution time: %s seconds\ntimes: {time_s}\nclock num: {clock}" % (end_time - start_time))
    a = 1
    mapping_display(ffn_fusion.best_mapping)
    clock = ffn_fusion.simulate(ffn_fusion.best_mapping, system.device)
    print(f"clock: {clock}")




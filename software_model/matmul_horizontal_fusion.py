from typing import List
from software_model.mutmul_fusion import MatmulFusion
from software_model.Fusion import Fusion
from software_model.horizontal_fusion import HorizontalFusion
from hardware_model.device import Device
from software_model.DataFrame import DataType, Tensor
from software_model.operators import Operator, Reshape, Transpose
from util.util import num_to_tile_list
from util.mapping import Mapping
from math import ceil
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy
class HorizontalMatmulFusion(HorizontalFusion):
    def __init__(self,fusion_list: List[Fusion], data_type: DataType):
        super().__init__(fusion_list, data_type)




    def compile_and_simulate(self, pcb_module: Device):
        min_cycle_count = 2**63 - 1
        best_mapping = None
        fusion_num = len(self.fusion_list)

        M = self.fusion_list[0].operator_list[0].M
        N = self.fusion_list[0].operator_list[0].N
        K = self.fusion_list[0].operator_list[0].K

        #TODO:注意llmcompass中M，N等于一时候矩阵乘法是利用vector_unit实现的
        l2_available_size = pcb_module.compute_module.l2_size

        for l2_tile_M in num_to_tile_list(6, 11, M):
            for l2_tile_N in num_to_tile_list(6, 11, N):
                if K <= 12288:
                    l2_K_tiling_factor_list = [1, 2, 4, 8]
                else:
                    l2_K_tiling_factor_list = [
                        K // 1024,
                        K // 2048,
                        K // 4096,
                        K // 8192,
                    ]
                for l2_K_tiling_factor in l2_K_tiling_factor_list:
                    l2_tile_K = ceil(
                        K / l2_K_tiling_factor
                    )
                    # we all use l2 double buffering
                    working_set_size =  self.fusion_list[0].buffer_store_cost([l2_tile_M, l2_tile_K, l2_tile_N]) * self.data_type.word_size * 2


                    available_tile_num = l2_available_size // working_set_size
                    if available_tile_num <= 0:
                        continue

                    for l1_tile_M in num_to_tile_list(5, 8, min(M, l2_tile_M)):
                        for l1_tile_N in num_to_tile_list(5, 8, min(N, l2_tile_N)):
                            for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                                l1_tile_K = ceil(l2_tile_K / l1_K_tiling_factor)
                                if (
                                    l1_tile_M * l1_tile_N
                                    + l1_tile_N * l1_tile_K
                                    + l1_tile_M * l1_tile_K
                                    > pcb_module.compute_module.core.SRAM_size
                                    // self.data_type.word_size
                                    // 2
                                ):#L1 use double buffer
                                    continue
                                #TODO: flash attention and ffn is kmn
                                
                                l2_loop_order = "knm"
                                l1_loop_order = "knm"
                                for (
                                        l0_M_tiling_factor,
                                        l0_N_tiling_factor,
                                        l0_K_tiling_factor,
                                ) in self.find_permutations(
                                    pcb_module.compute_module.core.systolic_array_count
                                ):
                                    mapping = Mapping(
                                        available_tile_num,
                                        l2_tile_M,
                                        l2_tile_N,
                                        l2_tile_K,
                                        l1_tile_M,
                                        l1_tile_N,
                                        l1_tile_K,
                                        l2_loop_order,
                                        l1_loop_order,
                                        l0_M_tiling_factor,
                                        l0_N_tiling_factor,
                                        l0_K_tiling_factor,
                                    )
                                    cycle_count = self.simulate(
                                        mapping,
                                        pcb_module,
                                    )
                                    if cycle_count < min_cycle_count:
                                        min_cycle_count = cycle_count
                                        best_mapping = mapping

        self.best_mapping = best_mapping
        # self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = min_cycle_count / pcb_module.compute_module.clock_freq
        # self.best_mapping.display()
        return self.latency

    def simulate(self, mapping: Mapping, pcb_module: Device)->int:
        # TODO: look up table not implemented
        M = self.fusion_list[0].operator_list[0].M
        N = self.fusion_list[0].operator_list[0].N
        K = self.fusion_list[0].operator_list[0].K
        data_type = self.data_type

        l2_tile_BS = mapping.l2_tile_BS
        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        assert (
                self.fusion_list[0].buffer_store_cost([l2_tile_M, l2_tile_K, l2_tile_N])
                <= pcb_module.compute_module.l2_size // data_type.word_size // 2)

        BS_l2_t = len(self.fusion_list) // l2_tile_BS
        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        BS_l2_t_remain = M % l2_tile_M
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K

        l2_tiles = np.empty(
            [ceil(len(self.fusion_list) / l2_tile_BS), ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )

    class L2TileSimulator:
        def __init__(
                self,
                BS:int,
                M: int,
                N: int,
                K: int,
                data_type: DataType,
                mapping: Mapping,
                pcb_module: Device,
                look_up_table: pd.DataFrame = None,
        ):
            self.BS = BS
            self.M = M
            self.N = N
            self.K = K
            self.K_reduction_cycle_count = ceil(
               BS * M * N / pcb_module.compute_module.total_vector_flops_per_cycle
            ) + 2 * ceil(
                BS
                * M
                * N
                * data_type.word_size
                / pcb_module.compute_module.l2_bandwidth_per_cycle
            )  # reduction只有沿K维度切才有用
            self.K_reduction_io_count = 2 * BS * M * N * data_type.word_size
            self.BS_M_K_io_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(BS * M * K, data_type, pcb_module.compute_module.clock_freq)
            self.BS_K_N_io_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(BS * K * N, data_type, pcb_module.compute_module.clock_freq)
            self.BS_M_N_io_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(BS * M * N, data_type, pcb_module.compute_module.clock_freq)
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(BS, M, N, K, data_type, mapping, pcb_module, look_up_table)

        def simulate_l2_tile_compute_cycle_count(
                self,
                BS:int,
                M:int,
                N:int,
                K:int,
                data_type:DataType,
                mapping:Mapping,
                chiplet_module:Device,
                look_up_table:pd.DataFrame = None
        )->int:

            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N
            l1_tile_K = mapping.l1_tile_K

            M_l1_t = M // l1_tile_M
            N_l1_t = N // l1_tile_N
            K_l1_t = K // l1_tile_K
            M_remain = M % l1_tile_M
            N_remain = N % l1_tile_N
            K_remain = K % l1_tile_K

            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=MatmulFusion.L1TileSimulator,
            )



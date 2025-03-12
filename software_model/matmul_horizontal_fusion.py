import logging
from typing import List
from software_model.mutmul_fusion import MatmulFusion
from software_model.Fusion import Fusion
from software_model.horizontal_fusion import HorizontalFusion
from hardware_model.device import Device
from software_model.matmul import Matmul, BatchedMatmul
from software_model.DataFrame import DataType, Tensor
from software_model.operators import Operator, Reshape, Transpose
from util.util import num_to_tile_list
from util.mapping import Mapping
from math import ceil, floor, log2
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy
from loguru import logger
class HorizontalMatmulFusion(HorizontalFusion):
    def __init__(self,fusion_list: List[Fusion], data_type: DataType):
        super().__init__(fusion_list, data_type)
        self.operator_list = fusion_list[0].operator_list
        self.look_up_table = None





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
                    l2_tile_K = 2 ** floor(log2(l2_tile_K))
                    working_set_size =  self.fusion_list[0].buffer_store_cost([l2_tile_M, l2_tile_K, l2_tile_N]) * self.data_type.word_size * 2


                    available_tile_num = l2_available_size // working_set_size
                    if available_tile_num <= 0:
                        continue

                    for l1_tile_M in num_to_tile_list(5, 8, min(M, l2_tile_M)):
                        for l1_tile_N in num_to_tile_list(5, 8, min(N, l2_tile_N)):
                            for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                                l1_tile_K = ceil(l2_tile_K / l1_K_tiling_factor)
                                if (
                                    self.fusion_list[0].buffer_store_cost([l1_tile_M, l1_tile_K, l1_tile_N])
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
                                        # logger.info(f"New best mapping found: {min_cycle_count}")

        self.best_mapping = best_mapping
        # self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = min_cycle_count / pcb_module.compute_module.clock_freq
        # self.best_mapping.display()
        return self.latency

    def simulate(self, mapping: Mapping, pcb_module: Device)->int:
        if self.look_up_table is None:
            self.look_up_table = pd.read_csv(
                f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
                header=None,
                names=[
                    "M",
                    "N",
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )  # 删除重复项
            # self.look_up_table.reset_index(drop=True, inplace=True)
            # self.look_up_table.to_csv(
            #     f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
            #     header=False,
            #     index=False,
            # )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
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
        BS_remain =  len(self.fusion_list) % l2_tile_BS
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K
        assert (M_l2_t > 0 and N_l2_t > 0 and K_l2_t > 0)
        l2_tiles = np.empty(
            [ceil(len(self.fusion_list) / l2_tile_BS), ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )
        if BS_l2_t * M_l2_t * N_l2_t * K_l2_t != 0:
            l2_tiles[:BS_l2_t, :M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_BS,
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if BS_remain != 0:
            l2_tiles[-1, :M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                BS_remain,
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if M_remain != 0:
            l2_tiles[:BS_l2_t, -1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_BS,
                M_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if N_remain != 0:
            l2_tiles[:BS_l2_t, :M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_BS,
                l2_tile_M,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if K_remain != 0:
            l2_tiles[:BS_l2_t, :M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_BS,
                l2_tile_M,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if BS_remain * M_remain !=0:
            l2_tiles[-1, -1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                BS_remain,
                M_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if BS_remain * N_remain != 0:
            l2_tiles[-1, :M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                BS_remain,
                l2_tile_M,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if BS_remain * K_remain != 0:
            l2_tiles[-1, :M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                BS_remain,
                l2_tile_M,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain != 0:
            l2_tiles[:BS_l2_t, -1, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_BS,
                M_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain != 0:
            l2_tiles[:BS_l2_t, -1, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_BS,
                M_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain != 0:
            l2_tiles[:BS_l2_t, :M_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_BS,
                l2_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if BS_remain * M_remain * N_remain != 0:
            l2_tiles[-1, -1, -1, :K_l2_t] = self.L2TileSimulator(
                BS_remain,
                M_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if BS_remain * M_remain * K_remain != 0:
            l2_tiles[-1, -1, :N_l2_t, -1] = self.L2TileSimulator(
                BS_remain,
                M_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if BS_remain * N_remain * K_remain != 0:
            l2_tiles[-1, :M_l2_t, -1, -1] = self.L2TileSimulator(
                BS_remain,
                l2_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain != 0:
            l2_tiles[:BS_l2_t, -1, -1, -1] = self.L2TileSimulator(
                l2_tile_BS,
                M_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )
        if BS_remain * M_remain * N_remain * K_remain != 0:
            l2_tiles[-1, -1, -1, -1] = self.L2TileSimulator(
                BS_remain,
                M_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                self.operator_list,
                pcb_module,
                self.look_up_table,
            )

        total_cycle_count = 0
        total_cycle_count += (
                l2_tiles[0,0, 0, 0].BS_M_K_io_cycle_count + l2_tiles[0, 0, 0, 0].BS_K_N_io_cycle_count
        )

        previous_bs = 0
        previous_m = 0
        previous_n = 0
        previous_k = 0

        for bs in range(ceil(len(self.fusion_list) / l2_tile_BS)):
            for m, n, k in Matmul.generate_tile_loops(
                ceil(M / l2_tile_M),
                ceil(N / l2_tile_N),
                ceil(K / l2_tile_K),
                mapping.l2_loop_order,
            ):
                current_tile = l2_tiles[bs, m, n, k]
                if bs == 0 and m == 0 and n == 0 and k == 0:
                    continue

                l2_tile = l2_tiles[bs, m, n, k]
                previous_l2_tile = l2_tiles[previous_bs, previous_m, previous_n, previous_k]

                # current tile read latency

                if bs == previous_bs and m == previous_m and k == previous_k:
                    current_tile_read_cycle_count = l2_tile.BS_M_K_io_cycle_count
                elif bs == previous_bs and k == previous_k and n == previous_n:
                    current_tile_read_cycle_count = l2_tile.BS_K_N_io_cycle_count
                else:
                    current_tile_read_cycle_count = (
                            l2_tile.BS_M_K_io_cycle_count + l2_tile.BS_K_N_io_cycle_count
                    )

                # if k > 0 and not (m == previous_m and n == previous_n and bs == previous_bs):
                #     current_tile_read_cycle_count += l2_tile.BS_M_N_io_cycle_count

                previous_tile_compute_cycle_count = previous_l2_tile.compute_cycle_count

                if m == previous_m and n == previous_n and bs == previous_bs:
                    previous_tile_write_cycle_count = 0
                else:
                    previous_tile_write_cycle_count = previous_l2_tile.BS_M_N_io_cycle_count

                total_cycle_count += (
                        max(
                            current_tile_read_cycle_count, previous_tile_compute_cycle_count
                        )
                        + previous_tile_write_cycle_count
                )
                previous_m = m
                previous_n = n
                previous_k = k
        # compute and write last tile
        total_cycle_count += (
                l2_tiles[-1, -1, -1, -1].BS_M_N_io_cycle_count
                + l2_tiles[-1, -1, -1, -1].compute_cycle_count
        )
        return total_cycle_count






    class L2TileSimulator:
        def __init__(
                self,
                BS:int,
                M: int,
                N: int,
                K: int,
                data_type: DataType,
                mapping: Mapping,
                operator_list: List[Operator],
                pcb_module: Device,
                look_up_table: pd.DataFrame = None,
        ):
            self.BS = BS
            self.M = M
            self.N = N
            self.K = K
            self.operator_list = operator_list
            self.BS_K_reduction_cycle_count = ceil(
               BS * M * N / pcb_module.compute_module.total_vector_flops_per_cycle
            ) + 2 * ceil(
                BS
                * M
                * N
                * data_type.word_size
                / pcb_module.compute_module.l2_bandwidth_per_cycle
            )  # reduction只有沿K维度切才有用
            self.BS_K_reduction_io_count = 2 * BS * M * N * data_type.word_size
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
            total_cycle_count = 0
            for operator in self.operator_list:
                if operator.__class__.__name__ == "Matmul":
                    l1_tiles = np.empty(
                        [BS,ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                        dtype=Matmul.L1TileSimulator,
                    )
                    if M_l1_t * N_l1_t * K_l1_t != 0:
                        l1_tiles[:, :M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                            l1_tile_M,
                            l1_tile_N,
                            l1_tile_K,
                            data_type,
                            mapping,
                            chiplet_module,
                            look_up_table,
                        )
                    if M_remain != 0:
                        l1_tiles[:, -1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                            M_remain,
                            l1_tile_N,
                            l1_tile_K,
                            data_type,
                            mapping,
                            chiplet_module,
                            look_up_table,
                        )
                    if N_remain != 0:
                        l1_tiles[:, :M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                            l1_tile_M,
                            N_remain,
                            l1_tile_K,
                            data_type,
                            mapping,
                            chiplet_module,
                            look_up_table,
                        )
                    if K_remain != 0:
                        l1_tiles[:, :M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                            l1_tile_M,
                            l1_tile_N,
                            K_remain,
                            data_type,
                            mapping,
                            chiplet_module,
                            look_up_table,
                        )
                    if M_remain * N_remain != 0:
                        l1_tiles[:, -1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                            M_remain,
                            N_remain,
                            l1_tile_K,
                            data_type,
                            mapping,
                            chiplet_module,
                            look_up_table,
                        )
                    if M_remain * K_remain != 0:
                        l1_tiles[:, -1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                            M_remain,
                            l1_tile_N,
                            K_remain,
                            data_type,
                            mapping,
                            chiplet_module,
                            look_up_table,
                        )
                    if N_remain * K_remain != 0:
                        l1_tiles[:, M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                            l1_tile_M,
                            N_remain,
                            K_remain,
                            data_type,
                            mapping,
                            chiplet_module,
                            look_up_table,
                        )
                    if M_remain * N_remain * K_remain != 0:
                        l1_tiles[:, -1, -1, -1] = Matmul.L1TileSimulator(
                            M_remain,
                            N_remain,
                            K_remain,
                            data_type,
                            mapping,
                            chiplet_module,
                            look_up_table,
                        )
                    M_K_tile_size = np.zeros(
                        [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=int
                    )

                    M_K_tile_size[:M_l1_t, :K_l1_t] = l1_tile_M * l1_tile_K
                    if M_remain > 0:
                        M_K_tile_size[-1, :K_l1_t] = M_remain * l1_tile_K
                    if K_remain > 0:
                        M_K_tile_size[:M_l1_t, -1] = l1_tile_M * K_remain
                    if M_remain > 0 and K_remain > 0:
                        M_K_tile_size[-1, -1] = M_remain * K_remain

                    K_N_tile_size = np.zeros(
                        [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=int
                    )
                    K_N_tile_size[:K_l1_t, :N_l1_t] = l1_tile_K * l1_tile_N
                    if K_remain > 0:
                        K_N_tile_size[-1, :N_l1_t] = K_remain * l1_tile_N
                    if N_remain > 0:
                        K_N_tile_size[:K_l1_t, -1] = l1_tile_K * N_remain
                    if K_remain > 0 and N_remain > 0:
                        K_N_tile_size[-1, -1] = K_remain * N_remain

                    M_N_tile_size = np.zeros(
                        [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=int
                    )
                    M_N_tile_size[:M_l1_t, :N_l1_t] = l1_tile_M * l1_tile_N
                    if M_remain > 0:
                        M_N_tile_size[-1, :N_l1_t] = M_remain * l1_tile_N
                    if N_remain > 0:
                        M_N_tile_size[:M_l1_t, -1] = l1_tile_M * N_remain
                    if M_remain > 0 and N_remain > 0:
                        M_N_tile_size[-1, -1] = M_remain * N_remain


                    previous_batch_Read_BS_M_K = np.zeros(
                        [BS, ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
                    )
                    previous_batch_Read_BS_K_N = np.zeros(
                        [BS, ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
                    )
                    previous_batch_Read_BS_M_N = np.zeros(
                        [BS, ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                    )
                    previous_batch_Write_BS_M_N = np.zeros(
                        [BS, ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                    )
                    previous_batch_compute_cycle_count = 0
                    active_l1_tile_list = []

                    current_BS = -1
                    for bs in range(BS):
                        for m,n,k in Matmul.generate_tile_loops(
                            ceil(M / l1_tile_M),
                            ceil(N / l1_tile_N),
                            ceil(K / l1_tile_K),
                            mapping.l1_loop_order
                        ):
                            active_l1_tile_list.append((bs, m, n, k, l1_tiles[bs, m, n, k]))
                            if (
                                    bs == BS - 1
                                    and m == ceil(M / l1_tile_M) - 1
                                    and n == ceil(N / l1_tile_N) - 1
                                    and k == ceil(K / l1_tile_K) - 1
                            ):
                                pass
                            elif (
                                    len(active_l1_tile_list) < chiplet_module.compute_module.core_count
                            ):
                                continue
                            assert (len(active_l1_tile_list) <= chiplet_module.compute_module.core_count)

                            current_batch_Read_BS_M_K = np.zeros(
                                [BS, ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
                            )
                            current_batch_Read_BS_K_N = np.zeros(
                                [BS, ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
                            )
                            current_batch_Read_BS_M_N = np.zeros(
                                [BS, ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                            )
                            current_batch_Write_BS_M_N = np.zeros(
                                [BS, ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                            )
                            current_batch_compute_cycle_count = 0
                            for i in range(len(active_l1_tile_list)):
                                temp_bs, temp_m, temp_n, temp_k, temp_l1_tile = active_l1_tile_list[i]
                                current_batch_Read_BS_M_K[temp_bs, temp_m, temp_k] = 1
                                current_batch_Read_BS_K_N[temp_bs, temp_k, temp_n] = 1
                                current_batch_Read_BS_M_N[temp_bs, temp_m, temp_n] = temp_k > 0
                                current_batch_Write_BS_M_N[temp_bs, temp_m, temp_n] = 1
                                temp_l1_tile_compute_cycle_count = temp_l1_tile.compute_cycle_count
                                if temp_k > 0: # OS dataflow
                                    temp_l1_tile_compute_cycle_count += ceil(
                                         temp_l1_tile.M
                                        * temp_l1_tile.N
                                        / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                                    )
                                current_batch_compute_cycle_count = max(
                                    current_batch_compute_cycle_count,
                                    temp_l1_tile_compute_cycle_count,
                                )

                            current_batch_BS_M_K_read_count = np.sum(
                                current_batch_Read_BS_M_K * (~previous_batch_Read_BS_M_K)
                                *M_K_tile_size
                            )
                            current_batch_BS_K_N_read_count = np.sum(
                                current_batch_Read_BS_K_N * (~previous_batch_Read_BS_K_N)
                                *K_N_tile_size
                            )
                            current_batch_BS_M_N_read_count = np.sum(
                                current_batch_Read_BS_M_N
                                * (~(previous_batch_Read_BS_M_N + previous_batch_Write_BS_M_N))
                                *M_N_tile_size
                            )
                            previous_batch_BS_M_N_write_count = np.sum(
                                (previous_batch_Write_BS_M_N * (~current_batch_Read_BS_M_N))
                                * M_N_tile_size
                            )
                            # read size
                            current_batch_read_count = (
                                    current_batch_BS_M_K_read_count
                                    + current_batch_BS_K_N_read_count
                                    + current_batch_BS_M_N_read_count
                            )

                            current_batch_read_cycle_count = ceil(
                                current_batch_read_count
                                * chiplet_module.compute_module.core.systolic_array.input_word_size
                                / chiplet_module.compute_module.l2_bandwidth_per_cycle
                            )
                            previous_batch_write_cycle_count = ceil(
                                previous_batch_BS_M_N_write_count
                                * chiplet_module.compute_module.core.systolic_array.output_word_size
                                / chiplet_module.compute_module.l2_bandwidth_per_cycle
                            )

                            total_cycle_count += (
                                    max(
                                        current_batch_read_cycle_count,
                                        previous_batch_compute_cycle_count,
                                    )
                                    + previous_batch_write_cycle_count
                            )

                            previous_batch_compute_cycle_count = current_batch_compute_cycle_count
                            previous_batch_Read_BS_M_K = copy.deepcopy(current_batch_Read_BS_M_K)
                            previous_batch_Read_BS_K_N = copy.deepcopy(current_batch_Read_BS_K_N)
                            previous_batch_Read_BS_M_N = copy.deepcopy(current_batch_Read_BS_M_N)
                            previous_batch_Write_BS_M_N = copy.deepcopy(current_batch_Write_BS_M_N)

                            active_l1_tile_list = []

                    total_cycle_count += previous_batch_compute_cycle_count + ceil(
                        np.sum(previous_batch_Write_BS_M_N * M_N_tile_size)
                        * data_type.word_size
                        / chiplet_module.compute_module.l2_bandwidth_per_cycle
                    )

                elif operator.__class__.__name__ == "Transpose":
                    total_cycle_count += ceil(
                        BS * M * N * data_type.word_size
                        / chiplet_module.compute_module.l2_bandwidth_per_cycle
                    )

                elif operator.__class__.__name__ == "Reshape":
                    pass

                else:
                    raise NotImplementedError(
                        f"operator {operator.__class__.__name__} is not supported")

            return total_cycle_count




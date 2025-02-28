from software_model.DataFrame import Tensor, DataType, size
from typing import List
from hardware_model.device import Device
from software_model.operators import Operator
from math import ceil, log2, floor
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy
from util.mapping import Mapping

class BatchedMatmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None

    def __call__(self, input1:Tensor, input2: Tensor) ->Tensor:
        # [bs, M, K] * [bs, K, N] = [bs, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        assert input1.shape[0] % input2.shape[0] == 0
        assert input1.shape[-1] == input2.shape[-2]
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.bs1 = input1.shape[0]
        self.bs2 = input2.shape[0]
        self.M = input1.shape[1]
        self.K = input1.shape[-1]
        self.N = input2.shape[-1]
        self.output_shape = input1.shape[:1] + input1.shape[1:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)
        return output


class Matmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None

    def __call__(self, input1:Tensor, input2: Tensor) ->Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        assert input1.shape[-1] == input2.shape[0]
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(input1.shape[:-1])
        self.K = input1.shape[-1]
        self.N = input2.shape[-1]

        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = 0 #Obtained during the calculation process
        return output


    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    # class Mapping:
    #     def __init__(
    #         self,
    #         l2_tile_M: int,
    #         l2_tile_N: int,
    #         l2_tile_K: int,
    #         is_l2_double_buffering: bool,
    #         l1_tile_M: int,
    #         l1_tile_N: int,
    #         l1_tile_K: int,
    #         l2_loop_order: str,
    #         l1_loop_order: str,
    #         l0_M_tiling_factor: int,
    #         l0_N_tiling_factor: int,
    #         l0_K_tiling_factor: int,
    #         dataflow: str = "os",
    #     ):
    #         self.l2_tile_M = l2_tile_M
    #         self.l2_tile_N = l2_tile_N
    #         self.l2_tile_K = l2_tile_K
    #         self.is_l2_double_buffering = is_l2_double_buffering
    #         self.l1_tile_M = l1_tile_M
    #         self.l1_tile_N = l1_tile_N
    #         self.l1_tile_K = l1_tile_K
    #         self.l2_loop_order = l2_loop_order
    #         self.l1_loop_order = l1_loop_order
    #         self.l0_M_tiling_factor = l0_M_tiling_factor
    #         self.l0_N_tiling_factor = l0_N_tiling_factor
    #         self.l0_K_tiling_factor = l0_K_tiling_factor
    #         self.dataflow = dataflow
    #
    #     def display(self):
    #         print(f'{"-" * 10} Mapping {"-" * 10}')
    #         print(
    #             f"l2_tile_M: {self.l2_tile_M}, l2_tile_N: {self.l2_tile_N}, l2_tile_K: {self.l2_tile_K}, is_l2_double_buffering: {self.is_l2_double_buffering}, l2_loop_order: {self.l2_loop_order}"
    #         )
    #         print(
    #             f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, l1_tile_K: {self.l1_tile_K}, l1_loop_order: {self.l1_loop_order}"
    #         )
    #         print(
    #             f"l0_M_tiling_factor: {self.l0_M_tiling_factor}, l0_N_tiling_factor: {self.l0_N_tiling_factor}, l0_K_tiling_factor: {self.l0_K_tiling_factor}"
    #         )
    def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: Mapping,
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
    ) -> int:
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
            dtype=Matmul.L1TileSimulator,
        )


    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: Mapping,
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L1 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, K, data_type, mapping, chiplet_module, look_up_table
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            assert (
                M * K + K * N + M * N
                <= chiplet_module.compute_module.core.SRAM_size
                // data_type.word_size
                // 2
            )

            M_tiling_factor = mapping.l0_M_tiling_factor
            N_tiling_factor = mapping.l0_N_tiling_factor
            K_tiling_factor = mapping.l0_K_tiling_factor
            assert (
                M_tiling_factor * K_tiling_factor * N_tiling_factor
                <= chiplet_module.compute_module.core.systolic_array_count
            )

            compute_cycle_count = ceil(
                Matmul.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    ceil(K / K_tiling_factor),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    chiplet_module.compute_module.core.systolic_array.mac_per_cycle,
                    mapping.dataflow,
                )
                + (K_tiling_factor - 1)
                * M
                * N
                / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )

            return compute_cycle_count
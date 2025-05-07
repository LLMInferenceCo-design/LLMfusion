from software_model.operators import Operator,Transpose,Concat, Reshape
from software_model.DataFrame import DataType, Tensor, size
from software_model.matmul import Matmul, BatchedMatmul
from typing import List
from functools import reduce
from software_model.Fusion import Fusion

from hardware_model.device import Device
from util.mapping import Mapping
import pandas as pd
import numpy as np
from math import ceil
class MatmulFusion(Fusion):
    def __init__(self, operator_list: List, data_type: DataType):
        super().__init__(operator_list, data_type)
        self.l2_loop_order = "knm"
        self.l1_loop_order = "knm"

    def buffer_store_cost(self,tile_size:List[int]):
        cost= 0
        for operator in self.operator_list:
             # tile_size = [M, K, N]
            if operator.__class__.__name__ in ("Matmul", "BatchedMatmul"):
                cost += (tile_size[0] * tile_size[1] + tile_size[1] * tile_size[2] + tile_size[0] * tile_size[2])
            # tile_size = [bs, M, K, N]
            elif operator.__class__.__name__ in ("Reshape", "Transpose","GeLU"):
                continue
            else:
                raise NotImplementedError(f"Unsupported operator {operator.__class__.__name__}")

        return cost

    def L2TileSimulator(
            self,
            mapping:Mapping,
            data_type: DataType,
            chiplet_module:Device,
            look_up_table:pd.DataFrame = None
    ):
        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        l1_tile_M = mapping.l1_tile_M
        l1_tile_N = mapping.l1_tile_N
        l1_tile_K = mapping.l1_tile_K

        M_l1_t = l2_tile_M // l1_tile_M
        N_l1_t = l2_tile_N // l1_tile_N
        K_l1_t = l2_tile_K // l1_tile_K



        for operator in self.operator_list:
            if operator.__class__.__name__ == "Reshape":
                continue
            elif operator.__class__.__name__ == "Transpose":
                continue

    class L1TileSimulator:
        def __init__(
            self,
            M:int,
            N:int,
            K:int,
            data_type:DataType,
            mapping:Mapping,
            chiplet_module: Device,
            operator_list: List[Operator],
            look_up_table:pd.DataFrame = None
        ):
            self.M = M
            self.K = K
            self.N = N
            self.compute_cycle_count = MatmulFusion.simulate_l1_tile_compute_cycle_count(
                            M, N, K, data_type, mapping, chiplet_module, look_up_table
                        )

        def simulate_l1_tile_compute_cycle_count(
                self,
                M,
                N,
                K,
                data_type,
                mapping:Mapping,
                chiplet_module:Device,
                look_up_table:pd.DataFrame):
            assert (MatmulFusion.buffer_store_cost([M, K, N])
             <= chiplet_module.compute_module.core.SRAM_size//data_type.word_size//2)

            for operator in self.operator_list:
                if operator.__class__.__name__ in ("Matmul", "BatchedMatmul"):
                    l1_tile_M = mapping.l1_tile_M
                    l1_tile_N = mapping.l1_tile_N
                    l1_tile_K = mapping.l1_tile_K
                    l2_tile_M = mapping.l2_tile_M
                    l2_tile_N = mapping.l2_tile_N
                    l2_tile_K = mapping.l2_tile_K
                    l2_tile_BS = mapping




    def read_io_cycle_count(self, l2_tile_BS, l2_tile_M, l2_tile_K, l2_tile_N, data_type, pcb_module:Device):
        return (pcb_module.io_module.simulate_l2_tile_io_cycle_count(l2_tile_BS * l2_tile_M * l2_tile_K, data_type, pcb_module.compute_module.clock_freq)
                + pcb_module.io_module.simulate_l2_tile_io_cycle_count(l2_tile_BS * l2_tile_K * l2_tile_N, data_type, pcb_module.compute_module.clock_freq))

    def write_io_cycle_count(self, l2_tile_BS, l2_tile_M, l2_tile_N, data_type, pcb_module:Device):
        return pcb_module.io_module.simulate_l2_tile_io_cycle_count(l2_tile_BS * l2_tile_M * l2_tile_N, data_type, pcb_module.compute_module.clock_freq)




    # def l1_store_cost(self,m,n,k):
    #     pass
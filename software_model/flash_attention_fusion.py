from hardware_model.device import Device
from software_model.Fusion import Fusion
from software_model.DataFrame import DataType, Tensor, size
from software_model.operators import Operator, Reshape, Transpose
from software_model.matmul import Matmul, BatchedMatmul
from typing import List
from util.util import num_to_tile_list
from util.mapping import Mapping
import pandas as pd
import numpy as np
from math import ceil
class FlashAttentionFusion(Fusion):
    def __init__(self, operator_list: List, data_type: DataType):
        super().__init__(operator_list, data_type)

    def buffer_store_cost(self,tile_size:List[int]):
        Br = tile_size[0]
        Bc = tile_size[2]
        d = tile_size[1]
        cost = 0

        # Kj, Vj
        cost += (Bc * d + Bc *d)
        # Qi, Oi, li, mi
        cost+= ( Br * d + Br * d + Br + Br)
        # ~mij， Sij，~lij
        cost += (Br + Br*Bc + Br)
        return cost

    def compile_and_simulate(self, pcb_module: Device):
        min_cycle_count = 2*63 - 1
        # only support MHA
        BS = self.operator_list[0].bs1
        M = self.operator_list[0].M
        K = self.operator_list[0].K
        N = self.operator_list[0].N

        l2_available_size = pcb_module.compute_module.l2_size
        for l2_tile_M in num_to_tile_list(8, 12, M):
            for l2_tile_N in num_to_tile_list(8, 12, N):
                l2_tile_K = K
                working_set_size = self.buffer_store_cost([l2_tile_M, l2_tile_K, l2_tile_N])
                available_tile_num = l2_available_size // working_set_size // self.data_type.word_size // 2
                if available_tile_num < 1:
                    continue

                for l1_tile_M in num_to_tile_list(5, 8, l2_tile_M):
                    for l1_tile_N in num_to_tile_list(5, 8, l2_tile_N):
                        l1_tile_K = l2_tile_K
                        if (
                            self.buffer_store_cost([l1_tile_M, l1_tile_K, l1_tile_N])
                            > pcb_module.compute_module.core.SRAM_size // self.data_type.word_size // 2
                        ):
                            continue

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
                                None,
                                None,
                                l0_M_tiling_factor,
                                l0_N_tiling_factor,
                                l0_K_tiling_factor,
                            )
                            cycle_count = self.simulate(mapping, pcb_module)
                            if cycle_count < min_cycle_count:
                                min_cycle_count = cycle_count
                                self.best_mapping = mapping
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

        M = self.operator_list[0].M
        K = self.operator_list[0].K
        N = self.operator_list[0].N
        BS = self.operator_list[0].bs1
        data_type = self.data_type

        l2_tile_BS = mapping.l2_tile_BS
        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        assert (
                self.buffer_store_cost([l2_tile_M, l2_tile_K, l2_tile_N])
                <= pcb_module.compute_module.l2_size // data_type.word_size // 2)

        BS_l2_t = BS // l2_tile_BS
        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N

        BS_remain = BS % l2_tile_BS
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N

        assert (M_l2_t > 0 and N_l2_t > 0)

        l2_tiles = np.empty(
            [ceil(BS / l2_tile_BS), ceil(M / l2_tile_M), ceil(N / l2_tile_N)],
            dtype=self.L2TileSimulator,
        )

    class L2TileSimulator:
        def __init__(
                self,
                BS:int,
                Br:int,
                Bc:int,
                K:int,
                data_type:DataType,
                mapping:Mapping,
                operator_list: List[Operator],
                pcb_module: Device,
                look_up_table:pd.DataFrame = None
        ):
            self.BS = BS
            self.M = Br
            self.N = Bc
            self.K = K
            self.operator_list = operator_list

            self.BS_K_V_read_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(
                BS * 2 *Bc * K,
                data_type,
                pcb_module.compute_module.clock_freq
            )
            self.BS_Q_read_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(
                BS * (Br * K + 2 * Br) ,
                data_type,
                pcb_module.compute_module.clock_freq
            )
            self.BS_O_l_m_read_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(
                BS * Br * K,
                data_type,
                pcb_module.compute_module.clock_freq
            )

            self.BS_O_l_m_write_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(
                BS * (Br * K + 2 * Br),
                data_type,
                pcb_module.compute_module.clock_freq
            )
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                BS, Br, Bc, K, data_type, mapping, pcb_module, look_up_table
            )

        def simulate_l2_tile_compute_cycle_count(
                self,
                BS:int,
                Br:int,
                Bc:int,
                K:int,
                data_type:DataType,
                mapping:Mapping,
                pcb_module:Device,
                look_up_table:pd.DataFrame = None
        ):

            l1_tile_Br = mapping.l1_tile_M
            l1_tile_Bc = mapping.l1_tile_N
            K = mapping.l1_tile_K

            Br_l1_t = Br // l1_tile_Br
            Bc_l1_t = Bc // l1_tile_Bc
            Br_remain = Br % l1_tile_Br
            Bc_remain = Bc % l1_tile_Bc
            total_cycle_count = 0
            l1_tiles = np.empty(
                [ceil(BS / l1_tile_Br), ceil(Br / l1_tile_Br), ceil(Bc / l1_tile_Bc)],
                dtype=FlashAttentionFusion.L1TileSimulator,
            )

            if Br_l1_t * Bc_l1_t != 0:
                l1_tiles[:, :Br_l1_t, :Bc_l1_t] = FlashAttentionFusion.L1TileSimulator(
                    l1_tile_Br,
                    l1_tile_Bc,
                    K,
                    data_type,
                    mapping,
                    self.operator_list,
                    pcb_module,
                    look_up_table,
                )
            if Br_remain != 0:
                l1_tiles[:, -1, :Bc_l1_t] = FlashAttentionFusion.L1TileSimulator(
                    Br_remain,
                    l1_tile_Bc,
                    K,
                    data_type,
                    mapping,
                    self.operator_list,
                    pcb_module,
                    look_up_table,
                )
            if Bc_remain != 0:
                l1_tiles[:, :Br_l1_t, -1] = FlashAttentionFusion.L1TileSimulator(
                    l1_tile_Br,
                    Bc_remain,
                    K,
                    data_type,
                    mapping,
                    self.operator_list,
                    pcb_module,
                    look_up_table,
                )
            if Br_remain != 0 and Bc_remain != 0:
                l1_tiles[:, -1, -1] = FlashAttentionFusion.L1TileSimulator(
                    Br_remain,
                    Bc_remain,
                    K,
                    data_type,
                    mapping,
                    self.operator_list,
                    pcb_module,
                    look_up_table,
                )



    class L1TileSimulator:
        def __init__(
                self,
                Br:int,
                Bc:int,
                K:int,
                data_type:DataType,
                mapping:Mapping,
                operator_list: List[Operator],
                chiplet_module: Device,
                look_up_table:pd.DataFrame = None
        ):
            assert (FlashAttentionFusion.buffer_store_cost([Br, K, Bc])
             <= chiplet_module.compute_module.core.SRAM_size//data_type.word_size//2)

            self.Br = Br
            self.Bc = Bc
            self.K = K
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                            Br, Bc, K, data_type, mapping, chiplet_module, look_up_table
                        )

        def simulate_l1_tile_compute_cycle_count(
                self,
                Br,
                Bc,
                K,
                data_type,
                mapping:Mapping,
                operator_list: List[Operator],
                chiplet_module:Device,
                look_up_table:pd.DataFrame
        ):
            flops_per_exp = chiplet_module.compute_module.core.vector_unit.flops_per_exp
            M_tiling_factor = mapping.l0_M_tiling_factor
            N_tiling_factor = mapping.l0_N_tiling_factor
            K_tiling_factor = mapping.l0_K_tiling_factor
            assert (
                    M_tiling_factor * K_tiling_factor * N_tiling_factor
                    <= chiplet_module.compute_module.core.systolic_array_count
            )
            compute_cycle_count = 0

            assert (
                operator_list[0].__class__.__name__ == 'BatchedMatmul' and
                operator_list[1].__class__.__name__ == 'Softmax' and
                operator_list[2].__class__.__name__ == 'BatchedMatmul'
            )

            for operator in operator_list:
                if operator.__class__.__name__ ==  'BatchedMatmul' and compute_cycle_count == 0:
                    # S = Q * KT
                    compute_cycle_count += (
                        Matmul.L1TileSimulator.simulate_l1_tile_compute_cycle_count(
                            Br, Bc, K, data_type, mapping, chiplet_module, look_up_table
                        )
                    )

                elif operator.__class__.__name__ ==  'BatchedMatmul':
                    # Pij * V
                    compute_cycle_count += (
                        Matmul.L1TileSimulator.simulate_l1_tile_compute_cycle_count(
                            Br, K, Bc, data_type, mapping, chiplet_module, look_up_table
                        )
                    )
                    # Oi
                    flops = 3 * Br * K
                    compute_cycle_count += ceil(
                        flops / chiplet_module.compute_module.core.vector_unit.flops_per_cycle
                    )

                elif operator.__class__.__name__ ==  'Softmax':
                    # ~mij， ~Pij， ~lij
                    flops = Br*Bc + Br * Bc * (flops_per_exp + 1) + Br * Bc
                    # mi_new, li_new,
                    flops += (Br + 2 * (2 * Br + Br * flops_per_exp))

                elif operator.__class__.__name__ in  ('Reshape', "Transpose"):
                    continue

                else:
                    raise NotImplementedError(f"Unsupported operator {operator.__class__.__name__}")

            return compute_cycle_count










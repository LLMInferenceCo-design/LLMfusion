
from software_model.Fusion import Fusion
from software_model.operators import DataType
from software_model.matmul import Matmul
from software_model.gelu import GeLU

from util.mapping import Mapping
from util.util import num_to_tile_list
from hardware_model.device import Device


import pandas as pd
import numpy as np
from math import ceil, floor, log2
from typing import List
from tqdm import tqdm
import time
class FFNFusion(Fusion):
    def __init__(self, operator_list: List, data_type: DataType):
        super().__init__(operator_list, data_type)
        assert (
            len(operator_list) == 3 and
            operator_list[0].__class__.__name__ == 'Matmul' and
            operator_list[-1].__class__.__name__ == 'Matmul' and
            operator_list[1].__class__.__name__ != 'Matmul' and
            operator_list[0].N == operator_list[-1].N

        )
    @staticmethod
    def buffer_store_cost(tile_size:List[int]):
        M, N, K, H = tile_size
        cost =M * K + K * N + M * N
        cost = max(cost, M * N + N * H + M * H)
        return cost

    # def time_display(self, i):
    #     finsh = "▓" * i
    #     need_do = "-" * (t - i)
    #     progress = (i / t) * 100
    #     dur = time.perf_counter() - start
    #     print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(progress, finsh, need_do, dur), end="")
    def compile_and_simulate(self, pcb_module: Device):
        min_cycle_count = 2 ** 63 - 1
        best_mapping = None

        M = self.operator_list[0].M
        N = self.operator_list[0].N
        K = self.operator_list[0].K
        H =self.operator_list[-1].N

        print("**************带时间的进度条**************")
        start = time.perf_counter()
        for l2_tile_M in tqdm(num_to_tile_list(6, 11, M)):
            for l2_tile_N in tqdm(num_to_tile_list(6, 11, N)):
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


                    l2_tile_H = l2_tile_K

                    working_set_size = self.buffer_store_cost([l2_tile_M, l2_tile_N, l2_tile_K, l2_tile_H]) * self.data_type.word_size * 2
                    if working_set_size > pcb_module.compute_module.l2_size:
                        continue

                    for l1_tile_M in num_to_tile_list(5, 8, l2_tile_M):
                        for l1_tile_N in num_to_tile_list(5, 8, l2_tile_N):
                            for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                                l1_tile_K = ceil(l2_tile_K / l1_K_tiling_factor)

                                l1_tile_H = l1_tile_K
                                if(
                                    self.buffer_store_cost([l1_tile_M, l1_tile_N, l1_tile_K, l1_tile_H])
                                    > pcb_module.compute_module.core.SRAM_size
                                    // self.data_type.word_size
                                    // 2
                                ):
                                    continue
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
                                        1,
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
                                        l2_tile_H,
                                        l1_tile_H,
                                    )
                                    cycle_count = self.simulate(
                                        mapping,
                                        pcb_module,
                                    )
                                    if cycle_count < min_cycle_count:
                                        min_cycle_count = cycle_count
                                        best_mapping = mapping
        self.best_mapping = best_mapping
        self.latency = min_cycle_count / pcb_module.compute_module.clock_freq
        return self.latency

    def simulate(self, mapping: Mapping, pcb_module: Device):
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
        N = self.operator_list[0].N
        K = self.operator_list[0].K
        H = self.operator_list[-1].N

        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K
        l2_tile_H = mapping.l2_tile_H

        assert (
            self.buffer_store_cost([l2_tile_M, l2_tile_N, l2_tile_K, l2_tile_H])
            <= pcb_module.compute_module.l2_size // self.data_type.word_size // 2
        )

        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        H_l2_t = H // l2_tile_H

        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K
        H_remain = H % l2_tile_H

        l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K), ceil(H / l2_tile_H)],
            dtype= self.L2TileSimulator
        )

        if M_l2_t * N_l2_t * K_l2_t * H_l2_t != 0:
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t, :H_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                l2_tile_H,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain!=0:
            l2_tiles[-1, :N_l2_t, :K_l2_t, :H_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                l2_tile_H,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain!=0:
            l2_tiles[:M_l2_t, -1, :K_l2_t, :H_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                l2_tile_H,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain!=0:
            l2_tiles[:M_l2_t, :N_l2_t, -1, :H_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                l2_tile_H,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if H_remain!=0:
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                H_remain,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain!=0:
            l2_tiles[-1, -1, :K_l2_t, :H_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                l2_tile_H,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain!=0:
            l2_tiles[-1, :N_l2_t, -1, :H_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                l2_tile_H,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * H_remain!=0:
            l2_tiles[-1, :N_l2_t, :K_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                H_remain,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain!=0:
            l2_tiles[:M_l2_t, -1, -1, :H_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                l2_tile_H,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * H_remain!=0:
            l2_tiles[:M_l2_t, -1, :K_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                H_remain,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain * H_remain!=0:
            l2_tiles[:M_l2_t, :N_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                H_remain,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain!=0:
            l2_tiles[-1, -1, -1, :H_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                l2_tile_H,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * H_remain!=0:
            l2_tiles[-1, -1, :K_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                H_remain,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain * H_remain!=0:
            l2_tiles[-1, :N_l2_t, -1, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                H_remain,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain * H_remain!=0:
            l2_tiles[:M_l2_t, -1, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                H_remain,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain * H_remain!=0:
            l2_tiles[-1, -1, -1, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                H_remain,
                self.data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        total_cycle_count = 0
        total_cycle_count += (
            l2_tiles[0, 0, 0, 0].M_K_io_cycle_count
            + l2_tiles[0, 0, 0, 0].K_N_io_cycle_count
        )
        previous_m = 0
        previous_n = 0
        previous_k = 0
        previous_h = 0

        for n in range(ceil(N / l2_tile_N)):
            for m in range(ceil(M / l2_tile_M)):
                for k in range(ceil(K / l2_tile_K)):
                    #TODO:
                    if m == 0 and n == 0 and k == 0:
                        continue

                    l2_tile = l2_tiles[m, n, k, previous_h]
                    previous_l2_tile = l2_tiles[previous_m, previous_n, previous_k, previous_h]

                    if m==previous_m and k==previous_k:
                        current_tile_read_cycle_count = l2_tile.M_K_io_cycle_count
                    elif k == previous_k and n == previous_n:
                        current_tile_read_cycle_count = l2_tile.K_N_io_cycle_count
                    else:
                        current_tile_read_cycle_count = l2_tile.M_K_io_cycle_count + l2_tile.K_N_io_cycle_count
                    #TODO:LLMcompass里面的reduction是不是要去掉
                    previous_tile_compute_cycle_count = previous_l2_tile.M_K_N_compute_cycle_count

                    if m == previous_m and n == previous_n:
                        previous_tile_write_cycle_count = 0
                    else:
                        previous_tile_write_cycle_count  = previous_l2_tile.M_H_io_cycle_count

                    total_cycle_count += (
                        max(current_tile_read_cycle_count, previous_tile_compute_cycle_count)
                        + previous_tile_write_cycle_count
                    )
                    previous_k = k



                for h in range(ceil(H / l2_tile_H)):
                    l2_tile = l2_tiles[m, n, previous_k, h]
                    previous_l2_tile = l2_tiles[previous_m, previous_n, previous_k, previous_h]

                    if m==previous_m and n==previous_n:
                        current_tile_read_cycle_count = l2_tile.N_H_io_cycle_count
                    elif h == previous_h and n == previous_n:
                        current_tile_read_cycle_count = l2_tile.M_N_io_cycle_count
                    else:
                        current_tile_read_cycle_count = l2_tile.M_N_io_cycle_count + previous_l2_tile.N_H_io_cycle_count

                    if h == 0:
                        previous_tile_compute_cycle_count = previous_l2_tile.M_K_N_compute_cycle_count + previous_l2_tile.M_N_Activation_compute_cycle_count
                    else:
                        previous_tile_compute_cycle_count = previous_l2_tile.M_N_H_compute_cycle_count

                    if m == previous_m and h == previous_h:
                        previous_tile_write_cycle_count = 0
                    else:
                        previous_tile_write_cycle_count = previous_l2_tile.M_H_io_cycle_count
                    total_cycle_count += (
                            max(current_tile_read_cycle_count, previous_tile_compute_cycle_count)
                            + previous_tile_write_cycle_count
                    )
                    previous_h = h
                    previous_n = n
                    previous_m = m
        total_cycle_count += (
            l2_tiles[-1, -1, -1, -1].M_N_H_compute_cycle_count + l2_tiles[-1, -1, -1, -1].M_H_io_cycle_count
        )
        if ceil(H / l2_tile_H) == 1:
            total_cycle_count += l2_tiles[-1, -1, -1, -1].M_N_Activation_compute_cycle_count
        # if ceil(M / l2_tile_M) == 1 and ceil(N / l2_tile_N) == 1 and ceil(K / l2_tile_K) == 1:
        #     total_cycle_count += l2_tiles[-1, -1, -1, -1].M_K_N_compute_cycle_count
        return total_cycle_count

    class L2TileSimulator:
        def __init__(
                self,
                M: int,
                N: int,
                K: int,
                H: int,
                data_type: DataType,
                mapping: Mapping,
                pcb_module: Device,
                look_up_table: pd.DataFrame = None
        ):
            self.M = M
            self.N = N
            self.K = K
            self.H = H
            self.M_K_io_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(M * K, data_type, pcb_module.compute_module.clock_freq)
            self.K_N_io_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(K * N, data_type, pcb_module.compute_module.clock_freq)
            self.N_H_io_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(N * H, data_type, pcb_module.compute_module.clock_freq)
            self.M_H_io_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(M * H, data_type, pcb_module.compute_module.clock_freq)
            self.M_N_io_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(M * N, data_type, pcb_module.compute_module.clock_freq)
            self.M_K_N_compute_cycle_count = Matmul.simulate_l2_tile_compute_cycle_count(M, N, K, data_type, mapping, pcb_module, look_up_table)
            self.M_N_H_compute_cycle_count = Matmul.simulate_l2_tile_compute_cycle_count(M, H, N, data_type, mapping(), pcb_module, look_up_table)
            self.M_N_Activation_compute_cycle_count = GeLU.simulate_l2_tile_compute_cycle_count(M * N, data_type, pcb_module)



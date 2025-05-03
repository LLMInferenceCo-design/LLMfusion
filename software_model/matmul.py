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
import os
from util.mapping import Mapping


class Matmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None
        self.l1_loop_order = "knm"

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

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str):
        assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(loop_M):
                for k in range(loop_K):
                    for n in range(loop_N):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(loop_N):
                for k in range(loop_K):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(loop_K):
                for n in range(loop_N):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(loop_K):
                for m in range(loop_M):
                    for n in range(loop_N):
                        yield m, n, k


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
    @staticmethod
    def simulate_l2_tile_compute_cycle_count(
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
        if M_l1_t * N_l1_t * K_l1_t != 0:
            l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                l1_tile_K,
                data_type,
                mapping,
                chiplet_module,
                look_up_table,
            )
        if M_remain != 0:
            l1_tiles[-1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                M_remain,
                l1_tile_N,
                l1_tile_K,
                data_type,
                mapping,
                chiplet_module,
                look_up_table,
            )
        if N_remain != 0:
            l1_tiles[:M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                l1_tile_M,
                N_remain,
                l1_tile_K,
                data_type,
                mapping,
                chiplet_module,
                look_up_table,
            )
        if K_remain != 0:
            l1_tiles[:M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                K_remain,
                data_type,
                mapping,
                chiplet_module,
                look_up_table,
            )
        if M_remain * N_remain != 0:
            l1_tiles[-1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                M_remain,
                N_remain,
                l1_tile_K,
                data_type,
                mapping,
                chiplet_module,
                look_up_table,
            )
        if M_remain * K_remain != 0:
            l1_tiles[-1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                M_remain,
                l1_tile_N,
                K_remain,
                data_type,
                mapping,
                chiplet_module,
                look_up_table,
            )
        if N_remain * K_remain != 0:
            l1_tiles[:M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                l1_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                chiplet_module,
                look_up_table,
            )
        if M_remain * N_remain * K_remain != 0:
            l1_tiles[-1, -1, -1] = Matmul.L1TileSimulator(
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

        total_cycle_count = 0
        previous_batch_Read_M_K = np.zeros(
            [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
        )
        previous_batch_Read_K_N = np.zeros(
            [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
        )
        previous_batch_Read_M_N = np.zeros(
            [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
        )
        previous_batch_Write_M_N = np.zeros(
            [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
        )
        previous_batch_compute_cycle_count = 0
        active_l1_tile_list = []
        for m, n, k in Matmul.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
                ceil(K / l1_tile_K),
                mapping.l1_loop_order,
        ):
            active_l1_tile_list.append((m, n, k, l1_tiles[m, n, k]))
            if (
                    m == ceil(M / l1_tile_M) - 1
                    and n == ceil(N / l1_tile_N) - 1
                    and k == ceil(K / l1_tile_K) - 1
            ):
                pass
            elif (
                    len(active_l1_tile_list) < chiplet_module.compute_module.core_count
            ):  # TODO:这里如果core_count不够，那不是没法处理
                continue

            assert (len(active_l1_tile_list) <= chiplet_module.compute_module.core_count), "core count not enough"
            current_batch_Read_M_K = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
            )
            current_batch_Read_K_N = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
            )
            current_batch_Read_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            current_batch_Write_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )

            current_batch_compute_cycle_count = 0
            for i in range(len(active_l1_tile_list)):
                temp_m, temp_n, temp_k, temp_l1_tile = active_l1_tile_list[i]
                current_batch_Read_M_K[temp_m, temp_k] = 1
                current_batch_Read_K_N[temp_k, temp_n] = 1
                current_batch_Read_M_N[temp_m, temp_n] = temp_k > 0
                current_batch_Write_M_N[temp_m, temp_n] = 1
                temp_l1_tile_compute_cycle_count = temp_l1_tile.compute_cycle_count
                if temp_k > 0:
                    temp_l1_tile_compute_cycle_count += ceil(
                        temp_l1_tile.M
                        * temp_l1_tile.N
                        / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                    )
                current_batch_compute_cycle_count = max(
                    current_batch_compute_cycle_count,
                    temp_l1_tile_compute_cycle_count,
                )

            # if one output tile in this batch shares input/output with another output tile in the previous batch, assign them to the same core to avoid data movement
            # note that of the three input matrix mk, kn, mn, at most one of them can be the same if we change m,n,k
            current_batch_M_K_read_count = np.sum(
                (current_batch_Read_M_K * (~previous_batch_Read_M_K))
                * M_K_tile_size
            )
            current_batch_K_N_read_count = np.sum(
                (current_batch_Read_K_N * (~previous_batch_Read_K_N))
                * K_N_tile_size
            )
            current_batch_M_N_read_count = np.sum(
                (
                        current_batch_Read_M_N
                        * (~(previous_batch_Read_M_N + previous_batch_Write_M_N))
                )
                * M_N_tile_size
            )
            previous_batch_M_N_write_count = np.sum(
                (previous_batch_Write_M_N * (~current_batch_Read_M_N))
                * M_N_tile_size
            )

            # read current batch while compute and write previous batch
            current_batch_read_count = (
                    current_batch_M_K_read_count
                    + current_batch_K_N_read_count
                    + current_batch_M_N_read_count
            )
            current_batch_read_cycle_count = ceil(
                current_batch_read_count
                * chiplet_module.compute_module.core.systolic_array.input_word_size
                / chiplet_module.compute_module.l2_bandwidth_per_cycle
            )
            prvious_batch_write_cycle_count = ceil(
                previous_batch_M_N_write_count
                * chiplet_module.compute_module.core.systolic_array.output_word_size
                / chiplet_module.compute_module.l2_bandwidth_per_cycle
            )

            total_cycle_count += (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
            )

            previous_batch_compute_cycle_count = current_batch_compute_cycle_count
            previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
            previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
            previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
            previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)

            active_l1_tile_list = []

        # last batch's compute and write
        total_cycle_count += previous_batch_compute_cycle_count + ceil(
            np.sum(previous_batch_Write_M_N * M_N_tile_size)
            * data_type.word_size
            / chiplet_module.compute_module.l2_bandwidth_per_cycle
        )

        return total_cycle_count

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
        @staticmethod
        def simulate_l1_tile_compute_cycle_count(

            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: Mapping,
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

    @staticmethod
    def simulate_systolic_array_cycle_count(
        look_up_table: pd.DataFrame,
        M,
        N,
        K,
        array_height,
        array_width,
        mac_per_clock,
        dataflow="os",
    ):
        # print(f'start: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        if M >= array_height and N >= array_width:
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.99
                )
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.98
                )
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        else:
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        # print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
            except KeyError:
                # print('not found in look up table')
                config = f"./systolic_array_model/temp/systolic_array_{os.getpid()}.cfg"
                os.makedirs(os.path.dirname(config), exist_ok=True)
                with open(config, "w") as f:
                    f.writelines("[general]\n")
                    f.writelines("run_name = systolic_array\n\n")
                    f.writelines("[architecture_presets]\n")
                    f.writelines("ArrayHeight:    " + str(array_height) + "\n")
                    f.writelines("ArrayWidth:     " + str(array_width) + "\n")
                    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
                    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("IfmapOffset:    0\n")
                    f.writelines("FilterOffset:   10000000\n")
                    f.writelines("OfmapOffset:    20000000\n")
                    f.writelines("Dataflow : " + dataflow + "\n")
                    f.writelines("Bandwidth : " + "100" + "\n")
                    f.writelines("MemoryBanks: 1\n\n")
                    f.writelines("[run_presets]\n")
                    f.writelines("InterfaceBandwidth: CALC\n")

                topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
                with open(topology, "w") as f:
                    f.writelines("Layer, M, N, K\n")
                    f.writelines(f"matmul1, {M}, {N}, {K},\n")

                logpath = f"./systolic_array_model/temp/"
                s = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=config,
                    topology=topology,
                    input_type_gemm=True,
                )
                s.run_scale(top_path=logpath)

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util
                with open(
                    f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv",
                    "a",
                ) as f:
                    f.writelines(
                        f"{M},{N},{K},{array_height},{array_width},{dataflow},{cycle_count},{util_rate:.3f}\n"
                    )

                look_up_table.loc[(M, N, K, array_height, array_width, dataflow), :] = [
                    cycle_count,
                    util_rate,
                ]
                if len(look_up_table) % 10 == 0:
                    look_up_table.sort_index(inplace=True)
        # if (
        #     dataflow == "os"
        # ):  # scalesim assumes collecting output is not on critical path in os
        #     cycle_count += min(array_height, array_width, M, N)
        # if True:
        #     print(f"{M}x{N}x{K}x{array_height}x{array_width}x{dataflow}: {cycle_count}")
        # new_table = look_up_table[~look_up_table.index.duplicated(keep='first')]
        # if look_up_table.shape[0]-new_table.shape[0]>=1:
        #     print(look_up_table)
        #     print(look_up_table.duplicated(keep=False))
        #     exit()
        # print(f'end: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        return ceil(cycle_count / mac_per_clock)

class BatchedMatmul(Matmul):
    def __init__(self, data_type: DataType):
        super().__init__(data_type)
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

    
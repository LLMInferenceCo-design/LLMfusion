from hardware_model.device import Device
from hardware_model.io_module import IOModule
from software_model.DataFrame import DataType, Tensor, size
from software_model.operators import Operator
import numpy as np
from math import ceil,log2

buffer_factor = 2

class LayerNorm(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input_shape = None

    def __call__(self, input: Tensor)->Tensor:
        self.input_shape = input.shape
        self.M = size(input.shape[:-1])
        self.N = input.shape[-1]
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )
        output = Tensor(input.shape, self.data_type)
        return output

    class ComputationalGraph:
        def __init__(self, M: int, N: int, data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

    class Mapping:
        def __init__(
                self,
                l2_tile_M: int,
                l2_tile_N: int,
                l1_tile_M: int,
                l1_tile_N: int,
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N

        def display(self):
            print("-" * 20)
            print(
                f"l2_tile_M: {self.l2_tile_M}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}"
            )
    def compile_and_simulate(self, pcb_module: Device, start_time = 0)->int:
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type

        l2_tile_N = N
        l2_tile_M = min( pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size) //2 //buffer_factor, M)

        l1_tile_N = N
        l1_tile_M = min( pcb_module.compute_module.core.SRAM_size // (l1_tile_N * data_type.word_size) //2 , l2_tile_M)

        while l1_tile_M < pcb_module.compute_module.core.vector_unit.vector_count and l1_tile_M != l2_tile_M:
            l1_tile_N = l1_tile_N // 2
            l1_tile_M = (
                    pcb_module.compute_module.core.SRAM_size
                    // (l1_tile_N * data_type.word_size)
                    // 2

            )

        mapping = self.Mapping(l2_tile_M, l2_tile_N, l1_tile_M, l1_tile_N)
        cycle_count = self.simulate(self.computational_graph, mapping, pcb_module)
        return cycle_count

    def simulate(self,
                 computational_graph: ComputationalGraph,
                 mapping: Mapping,
                 pcb_module: Device,
                 )->int:
        M = computational_graph.M
        N = computational_graph.N
        data_type = computational_graph.data_type
        l2_tile_M = mapping.l2_tile_M

        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.L2TileSimulator)

        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N,
                data_type,
                mapping,
                pcb_module,
            )
        if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
            )

        total_cycle_count = 0
        #TODO: double buffer，three types, for paper
        read_cycle = l2_tiles[0].read_cycle_count
        comp_cycle = l2_tiles[0].compute_cycle_count
        write_cycle = l2_tiles[0].write_cycle_count

        read_cycle_remain = l2_tiles[-1].read_cycle_count if M_remain != 0 else 0
        write_cycle_remain = l2_tiles[-1].write_cycle_count if M_remain != 0 else 0
        comp_cycle_remain = l2_tiles[-1].compute_cycle_count if M_remain != 0 else 0

        assert read_cycle == write_cycle and read_cycle_remain == write_cycle_remain , "read and write cycle should be the same"

        if comp_cycle >= read_cycle + write_cycle:
            total_cycle_count = comp_cycle * M_l2_t + read_cycle + max(comp_cycle_remain, write_cycle) + write_cycle_remain
        elif comp_cycle >= read_cycle_remain + write_cycle:
            if M_l2_t == 1:
                total_cycle_count = read_cycle + comp_cycle + max(comp_cycle_remain, write_cycle) + write_cycle_remain
            else:
                total_cycle_count = (read_cycle + comp_cycle + (M_l2_t - 2) * (read_cycle + write_cycle) + comp_cycle + max(comp_cycle_remain, write_cycle) + write_cycle_remain)

        elif comp_cycle >= read_cycle:
            total_cycle_count =(read_cycle + write_cycle) * (M_l2_t -1) + (M_l2_t > 1) * read_cycle_remain + (M_l2_t == 1) * read_cycle + max(comp_cycle_remain, write_cycle) + write_cycle_remain

        elif comp_cycle >= read_cycle_remain:
            total_cycle_count = (read_cycle + write_cycle) * M_l2_t + comp_cycle * (M_l2_t == 1) + read_cycle_remain * (M_l2_t > 1) + write_cycle_remain
        else:
            total_cycle_count = (read_cycle + write_cycle) * M_l2_t + read_cycle_remain + write_cycle_remain

        return total_cycle_count

    class L2TileSimulator:
        def __init__(self,
                   M:int,
                   N:int,
                   data_type: DataType,
                   mapping: "LayerNorm.Mapping",
                   pcb_module: Device):
            self.M = M
            self.N = N
            self.read_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count( M*N, data_type)
            self.write_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count( M*N, data_type)
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )

        def simulate_l2_tile_compute_cycle_count(self,
                                                 M: int,
                                                 N: int,
                                                 data_type: DataType,
                                                 mapping: "LayerNorm.Mapping",
                                                 pcb_module: Device)->int:
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = LayerNorm.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)

            l1_tile_cycle_count = (
                    l1_tile.read_cycle_count * 3
                    + l1_tile.write_cycle_count
                    + l1_tile.compute_cycle_count
            )
            total_cycle_count = ((ceil(l1_tile_count / pcb_module.compute_module.core_count)) *
                                 (l1_tile_cycle_count + (ceil(N / l1_tile_N) - 1) * (l1_tile.reduction_cycle_count)
                                ))
            return total_cycle_count

    class L1TileSimulator:
        def __init__(
                self,
                M: int,
                N: int,
                data_type: DataType,
                mapping: "LayerNorm.Mapping",
                pcb_module: Device,
        ):
            self.M = M
            self.N = N
            self.read_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(M*N, data_type)
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )
            self.write_cycle_count = pcb_module.io_module.simulate_l2_tile_io_cycle_count(M*N, data_type)
            self.reduction_cycle_count = (
                    M
                    * N
                    / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                    + M
                    * N
                    * data_type.word_size
                    * 2
                    / (
                            pcb_module.compute_module.l2_bandwidth_per_cycle
                            / pcb_module.compute_module.core_count
                    )
            )

        def simulate_l1_tile_compute_cycle_count(
                self,
                M: int,
                N: int,
                data_type: DataType,
                mapping: "LayerNorm.Mapping",
                pcb_module: Device,
        ):
            M_per_vector_count = ceil(
                M / pcb_module.compute_module.core.vector_unit.vector_count
            )
            N_per_vector_count = N
            M_per_vector_lane = M_per_vector_count
            N_per_vector_lane = ceil(
                N_per_vector_count
                / pcb_module.compute_module.core.vector_unit.vector_width
            )

            # each lane computes it own mean
            total_cycle_count = ceil(
                N_per_vector_lane
                * M_per_vector_lane
                / pcb_module.compute_module.core.vector_unit.flops_per_cycle
            )
            # the whole vector reduce to one mean
            total_cycle_count += log2(
                pcb_module.compute_module.core.vector_unit.vector_width
            )
            # each lane computes it own variance
            total_cycle_count += (
                    ceil(
                        N_per_vector_lane
                        * M_per_vector_lane
                        / pcb_module.compute_module.core.vector_unit.flops_per_cycle
                    )
                    * 2
            )
            # the whole vector reduce to one variance
            total_cycle_count += log2(
                pcb_module.compute_module.core.vector_unit.vector_width
            )
            # calculate normalized output
            total_cycle_count += (
                    ceil(
                        N_per_vector_lane
                        * M_per_vector_lane
                        / pcb_module.compute_module.core.vector_unit.flops_per_cycle
                    )
                    * 4
            )  # division is heavy

            return total_cycle_count
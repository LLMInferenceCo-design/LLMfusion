from software_model.DataFrame import Tensor, DataType, size
from software_model.operators import Operator
from hardware_model.device import Device
from util.mapping import Mapping
import pandas as pd
from math import ceil

class GeLU(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:])
        self.computational_graph = self.ComputationalGraph(
            self.M, self.data_type
        )
        return input

    class ComputationalGraph:
        def __init__(self, M: int, data_type: DataType):
            self.M = M
            self.data_type = data_type

    @staticmethod
    def simulate_l2_tile_compute_cycle_count(
            M: int,
            data_type: DataType,
            pcb_module: Device,

    ):
        parallelism = (
                pcb_module.compute_module.core_count
                * pcb_module.compute_module.core.vector_unit.vector_width
                * pcb_module.compute_module.core.vector_unit.vector_count
        )
        M = ceil(M / parallelism) * parallelism
        total_io_count = M * 2 * data_type.word_size
        io_cycle_count = ceil(total_io_count / pcb_module.compute_module.l2_bandwidth_per_cycle)
        total_flop_count = M * (10 + pcb_module.compute_module.core.vector_unit.flops_per_exp)
        compute_cycle_count = (
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                / pcb_module.compute_module.core_count
                / pcb_module.compute_module.clock_freq
        )

        return max(io_cycle_count, compute_cycle_count)

from software_model.operators import Operator
from software_model.DataFrame import DataType, Tensor, size
from hardware_model.device import Device
from typing import List
from functools import reduce
class Fusion(Operator):
    def __init__(self, operator_list: List, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.finish_flag = False
        self.operator_list = operator_list
        self.global_buffer_usage = 0
        self.core_usage = 0
        self.best_mapping = None
        self.look_up_table = None

    @staticmethod
    def find_permutations(n):
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations)


    def L1TileSimulator(self,):
        raise NotImplementedError("L1TileSimulator should be overridden in child classes")

    def buffer_store_cost(self,tile_size:List[int]):
        raise NotImplementedError("buffer_store_cost should be overridden in child classes")

    def read_io_cycle_count(self, l2_tile_BS, l2_tile_M, l2_tile_K, l2_tile_N, data_type, pcb_module:Device):
        raise NotImplementedError("This method should be overridden in child classes")


    def write_io_cycle_count(self, l2_tile_BS, l2_tile_M, l2_tile_N, data_type, pcb_module:Device):
        raise NotImplementedError("This method should be overridden in child classes")

    def compile_and_simulate(self, pcb_module: Device):
        raise NotImplementedError("compile_and_simulate method should be overridden in child classes")
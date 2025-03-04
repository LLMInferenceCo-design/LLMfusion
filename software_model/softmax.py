from software_model.DataFrame import Tensor, DataType, size
from software_model.operators import Operator
from util.mapping import Mapping
import numpy as np

class Softmax(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:-1])
        self.N = input.shape[-1]
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )
        return input

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu * 1e6}us")

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
                is_l2_double_buffering: bool,
                l1_tile_M: int,
                l1_tile_N: int,
                is_l1_double_buffering: bool = False,
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.is_l1_double_buffering = is_l1_double_buffering

        def display(self):
            print("-" * 20)
            print(
                f"l2_tile_M: {self.l2_tile_M}, is_l2_double_buffering: {self.is_l2_double_buffering}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, is_l1_double_buffering: {self.is_l1_double_buffering}"
            )

    # class L1TileSimulator:
    #     def __init__(self, mapping: Mapping, data_type: DataType):
    #         self.mapping = mapping
    #         self.data_type = data_type
    #
    #     def simulate(self, input: Tensor, output: Tensor):
    #         assert self.data_type == input.data_type
    #         assert self.data_type == output.data_type
    #         assert input.shape == output.shape
    #         assert input.shape == self.mapping.l2_tile_M * self.mapping.l2_tile_N
    #         assert input.shape == self.mapping.l1_tile_M * self.mapping.l1_tile_N
    #
    #         M = input.shape[0]
    #         N = input.shape[1]
    #
    #         for i in range(M):
    #             for j in range(N):
    #                 output[i, j] = input[i, j]
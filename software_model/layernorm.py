from hardware_model.device import Device
from software_model.DataFrame import DataType, Tensor, size
from software_model.operators import Operator

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

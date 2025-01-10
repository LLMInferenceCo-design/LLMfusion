from software_model.DataFrame import Tensor, DataType, size
from software_model.operators import Operator

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
from software_model.DataFrame import DataType,Tensor, size
from typing import Any, List

class CommunicationPrimitive:
    def __init__(self, data_type: DataType) -> None:
        self.data_type = data_type
        # simulation results
        self.latency = None

class AllReduceMultiPCB(CommunicationPrimitive):
    def __init__(self, data_type: DataType) -> None:
        super().__init__(data_type)

    def __call__(self, tensor: Tensor) -> Any:
        assert tensor.data_type == self.data_type
        self.input_shape = tensor.shape
        return tensor
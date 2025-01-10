from software_model.operators import Operator
from software_model.DataFrame import DataType, Tensor, size
from typing import List
class Operator_fusion(Operator):
    def __init__(self, operator_list: List, backward_layers: List, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.finish_flag = False
        self.operator_list = operator_list
        self.backward_layers = backward_layers


    def run_ready(self) -> bool:
        return all(layer.finish_flag for layer in self.backward_layers)
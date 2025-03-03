
from software_model.Fusion import Fusion
from software_model.DataFrame import DataType, Tensor, size
from software_model.matmul import Matmul, BatchedMatmul
from typing import List
class FlashAttentionFusion(Fusion):
    def __init__(self, operator_list: List, data_type: DataType):
        super().__init__(operator_list, data_type)

    def buffer_store_cost(self,tile_size:List[int]):
        pass
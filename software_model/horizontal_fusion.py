from typing import List
from software_model.mutmul_fusion import MatmulFusion
from software_model.Fusion import Fusion
from hardware_model.device import Device
from software_model.DataFrame import DataType, Tensor
from software_model.operators import Operator, Reshape, Transpose
from util.util import num_to_tile_list
from util.mapping import Mapping
from math import ceil
import numpy as np
import pandas as pd

class HorizontalFusion(Operator):
    def __init__(self, fusion_list:List[Fusion], data_type:DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.fusion_list = fusion_list
        self.data_type = data_type
        self.best_mapping = None


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



    def compile_and_simulate(self, pcb_module: Device):
        raise NotImplementedError("HorizontalFusion is an abstract class, please use its subclass compile_and_simulate")

    def simulate(self, mapping: Mapping, pcb_module: Device):
        raise NotImplementedError("HorizontalFusion is an abstract class, please use its subclass simulate")

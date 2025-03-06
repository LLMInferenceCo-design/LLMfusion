class Mapping:
    def __init__(
            self,
            available_tile_num: int,
            l2_tile_M: int,
            l2_tile_N: int,
            l2_tile_K: int,
            l1_tile_M: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l2_loop_order: str,
            l1_loop_order: str,
            l0_M_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int,
            l2_tile_H: int = 0, # 第二次矩阵乘分区
            l1_tile_H: int = 0, #
            dataflow: str = "os",
    ):
        self.l2_tile_BS = available_tile_num
        self.l2_tile_M = l2_tile_M
        self.l2_tile_N = l2_tile_N
        self.l2_tile_K = l2_tile_K
        self.l1_tile_M = l1_tile_M
        self.l1_tile_N = l1_tile_N
        self.l1_tile_K = l1_tile_K
        self.l2_loop_order = l2_loop_order
        self.l1_loop_order = l1_loop_order
        self.l0_M_tiling_factor = l0_M_tiling_factor
        self.l0_N_tiling_factor = l0_N_tiling_factor
        self.l0_K_tiling_factor = l0_K_tiling_factor
        self.l2_tile_H = l2_tile_H
        self.l1_tile_H = l1_tile_H
        self.dataflow = dataflow
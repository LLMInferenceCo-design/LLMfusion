from software_model.operators import DataType
from math import ceil
class IOModule:
    def __init__(self, bandwidth, latency):
        self.bandwidth = bandwidth
        self.latency = latency


    def simulate_l2_tile_io_cycle_count(self, l2_tile_size, data_type: DataType, clock_freq):
        return ceil( l2_tile_size * data_type.word_size / (self.bandwidth / clock_freq))


IO_module_dict = {
    "A100": IOModule(2039e9, 1e-6),
    "TPUv3": IOModule(float("inf"), 1e-6),
    "MI210": IOModule(1.6e12, 1e-6)
}

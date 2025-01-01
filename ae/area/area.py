from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
import json

with open("../../configs/GA100.json", "r") as f:
    configs_dict = json.load(f)

compute_die_area_mm2, A100_core_breakdown_map, compute_total_die_map = (
    calc_compute_chiplet_area_mm2(configs_dict, verbose=True)
)
io_die_area_mm2, io_total_die_map = calc_io_die_area_mm2(configs_dict, verbose=True)
print('hello')
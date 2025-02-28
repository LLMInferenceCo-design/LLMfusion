import json
from cost_model.cost_model import calc_compute_chiplet_area_mm2,calc_io_die_area_mm2

def template_to_system(arch_specs):
    device_specs = arch_specs["device"]
    compute_chiplet_specs = device_specs["compute_chiplet"]
    io_specs = device_specs["io"]
    core_specs = compute_chiplet_specs["core"]
    sublane_count = core_specs["sublane_count"]
    # vector unit
    vector_unit_specs = core_specs["vector_unit"]
    vector_unit = VectorUnit(
        sublane_count
        * vector_unit_specs["vector_width"]
        * vector_unit_specs["flop_per_cycle"],
        int(re.search(r"(\d+)", vector_unit_specs["data_type"]).group(1)) // 8,
        35,
        vector_unit_specs["vector_width"],
        sublane_count,
    )
    # systolic array
    systolic_array_specs = core_specs["systolic_array"]
    systolic_array = SystolicArray(
        systolic_array_specs["array_height"],
        systolic_array_specs["array_width"],
        systolic_array_specs["mac_per_cycle"],
        int(re.search(r"(\d+)", systolic_array_specs["data_type"]).group(1)) // 8,
        int(re.search(r"(\d+)", systolic_array_specs["data_type"]).group(1)) // 8,
    )
    # core
    core = Core(
        vector_unit,
        systolic_array,
        sublane_count,
        core_specs["SRAM_KB"] * 1024,
    )
    # compute module
    compute_module = ComputeModule(
        core,
        compute_chiplet_specs["core_count"] * device_specs["compute_chiplet_count"],
        device_specs["frequency_Hz"],
        io_specs["global_buffer_MB"] * 1024 * 1024,
        io_specs["global_buffer_bandwidth_per_cycle_byte"],
        overhead_dict["A100"],
    )
    # io module
    io_module = IOModule(
        io_specs["memory_channel_active_count"]
        * io_specs["pin_count_per_channel"]
        * io_specs["bandwidth_per_pin_bit"]
        // 8,
        1e-6,
    )
    # memory module
    memory_module = MemoryModule(
        device_specs["memory"]["total_capacity_GB"] * 1024 * 1024 * 1024
    )
    # device
    device = Device(compute_module, io_module, memory_module)
    # interconnect
    interconnect_specs = arch_specs["interconnect"]
    link_specs = interconnect_specs["link"]
    link_module = LinkModule(
        link_specs["bandwidth_per_direction_byte"],
        link_specs["bandwidth_both_directions_byte"],
        link_specs["latency_second"],
        link_specs["flit_size_byte"],
        link_specs["max_payload_size_byte"],
        link_specs["header_size_byte"],
    )
    interconnect_module = InterConnectModule(
        arch_specs["device_count"],
        TopologyType.FC
        if interconnect_specs["topology"] == "FC"
        else TopologyType.RING,
        link_module,
        interconnect_specs["link_count_per_device"],
    )

    # system
    system = System(device, interconnect_module)

    return system


def chage_hardware_params(params,config_file_path='../configs/GA100.json'):
    with open(config_file_path, "r") as f:
        arch_specs = json.load(f)
    
    arch_specs['device']['io']['global_buffer_MB'] = params['global_buffer_MB'] if 'global_buffer_size' in params else None
    arch_specs['device']['compute_chiplet']['core_count'] = params['core_count'] if 'core_count' in params else None
    arch_specs['device']['compute_chiplet']['core']['sublane_count'] = params['sublane_count'] if 'sublane_count' in params else None
    arch_specs['device']['compute_chiplet']['core']['systolic_array']['array_width'] = params['array_width'] if 'array_width' in params else None
    arch_specs['device']['compute_chiplet']['core']['systolic_array']['array_height'] = params['array_height'] if 'array_height' in params else None
    arch_specs['device']['compute_chiplet']['core']['vector_unit']['vector_width'] = params['vector_width'] if 'vector_width' in params else None
    arch_specs["device"]["compute_chiplet"]["core"]["SRAM_KB"] = params['SRAM_KB'] if 'SRAM_KB' in params else None
    
    # for area
    arch_specs["device"]["compute_chiplet"]["physical_core_count"] = arch_specs['device']['compute_chiplet']['core_count']
    arch_specs["device"]["compute_chiplet"]["core"]["vector_unit"]["int32_count"] = (
            arch_specs['device']['compute_chiplet']['core']["vector_unit"]['vector_width'] // 2
    )
    arch_specs["device"]["compute_chiplet"]["core"]["vector_unit"]["fp32_count"] = (
            arch_specs['device']['compute_chiplet']['core']["vector_unit"]['vector_width'] // 2
    )
    arch_specs["device"]["compute_chiplet"]["core"]["vector_unit"]["fp64_count"] = (
            arch_specs['device']['compute_chiplet']['core']["vector_unit"]['vector_width'] // 4
    )
    if arch_specs['device']['compute_chiplet']['core']["vector_unit"]['vector_width'] <= 32:
        arch_specs["device"]["compute_chiplet"]["core"]["register_file"][
            "num_registers"
        ] = (arch_specs['device']['compute_chiplet']['core']["vector_unit"]['vector_width'] * 512)
    else:
        arch_specs["device"]["compute_chiplet"]["core"]["register_file"][
            "num_reg_files"
        ] = (arch_specs['device']['compute_chiplet']['core']["vector_unit"]['vector_width'] // 32)
    compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
    io_area_mm2 = calc_io_die_area_mm2(arch_specs)

def num_to_tile_list(start_power, end_power, end_num):
   tile_list = []
   tem = 2**start_power
   end_list_value = min(end_num, 2 ** end_power)
   while tem < end_list_value:
       tile_list.append(tem)
       tem *= 2
   tile_list.append(end_list_value)
   # if tem >= end_num and tem <= 2**end_power:
   #     tile_list.append(end_list_value)
   return tile_list


if __name__ =="__main__":
    print(num_to_tile_list(6, 11, 5555)) # [2, 4, 8, 16, 32, 64, 100]
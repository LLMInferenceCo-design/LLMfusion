import json,re
from hardware_model.compute_module import (VectorUnit,SystolicArray,Core,ComputeModule,overhead_dict)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.device import Device
from hardware_model.interconnect import LinkModule, InterConnectModule, TopologyType
from hardware_model.system import System

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
    )#single core
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
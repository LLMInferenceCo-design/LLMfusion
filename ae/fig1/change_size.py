import json
from cost_model.cost_model import calc_compute_chiplet_area_mm2,calc_io_die_area_mm2
from design_space_exploration.dse import template_to_system

def chage_hardware_params(params, config_file_path='../configs/GA100.json'):
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
    hardware_system = t


import json
from cost_model.cost_model import calc_compute_chiplet_area_mm2,calc_io_die_area_mm2
from design_space_exploration.dse import template_to_system
import logging
from LLM_model.opt175b import opt175b_prefill
from software_model.DataFrame import DataType, Tensor, data_type_dict

def chage_hardware_params(params, arch_specs):

    arch_specs['device']['io']['global_buffer_MB'] = params.get('global_buffer_MB', arch_specs['device']['io']['global_buffer_MB'])
    arch_specs['device']['compute_chiplet']['core_count'] = params.get('core_count', arch_specs['device']['compute_chiplet']['core_count'])
    arch_specs['device']['compute_chiplet']['core']['sublane_count'] = params.get('sublane_count', arch_specs['device']['compute_chiplet']['core']['sublane_count'])
    arch_specs['device']['compute_chiplet']['core']['systolic_array']['array_width'] = params.get('array_width', arch_specs['device']['compute_chiplet']['core']['systolic_array']['array_width'])
    arch_specs['device']['compute_chiplet']['core']['systolic_array']['array_height'] = params.get('array_height', arch_specs['device']['compute_chiplet']['core']['systolic_array']['array_height'])
    arch_specs['device']['compute_chiplet']['core']['vector_unit']['vector_width'] = params.get('vector_width', arch_specs['device']['compute_chiplet']['core']['vector_unit']['vector_width'])
    arch_specs["device"]["compute_chiplet"]["core"]["SRAM_KB"] = params.get('SRAM_KB', arch_specs["device"]["compute_chiplet"]["core"]["SRAM_KB"])

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
    logging.info(f"compute_area_mm2: {compute_area_mm2}, io_area_mm2: {io_area_mm2}")
    hardware_system = template_to_system(arch_specs)

    return hardware_system



if __name__ == "__main__":
    hardware_config = {
        "array_width": 16,
        "array_height": 16,

    }
    input_seq_length = 2048
    batch_size = 8
    output_seq_length = 1024
    with open('../../configs/ga102_template.json', "r") as f:
        arch_specs = json.load(f)
    system = chage_hardware_params(hardware_config, arch_specs)
    prefill_model = opt175b_prefill(12288, 96, arch_specs['device_count'], data_type= data_type_dict['fp16'])
    _ = prefill_model(Tensor([batch_size, input_seq_length, prefill_model.d_model]))
    cycle_latency = prefill_model.compile_and_simulate(system)
    logging.info(f"cycle_latency: {cycle_latency}")
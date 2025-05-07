from software_model.DataFrame import DataType,Tensor, size,data_type_dict
from typing import Any, List
from hardware_model.device import Device
from hardware_model.interconnect import (
    LinkModule,
    InterConnectModule,
    TopologyType,
    interconnect_module_dict,
)
from math import ceil

class CommunicationPrimitive:
    def __init__(self, data_type: DataType) -> None:
        self.data_type = data_type
        # simulation results
        self.latency = None

class AllReduceMultiPCB(CommunicationPrimitive):
    def __init__(self, data_type: DataType) -> None:
        super().__init__(data_type)

    def __call__(self, tensor: Tensor) -> Any:
        assert tensor.data_type == self.data_type
        self.input_shape = tensor.shape
        return tensor
    
    def simulate(self, interconnect_module: InterConnectModule) -> None:
        device_count = interconnect_module.device_count
        link_bandwidth_per_direction = (
            interconnect_module.link_module.bandwidth_per_direction
        )
        link_bandwidth_both_direction = (
            interconnect_module.link_module.bandwidth_both_direction
        )
        link_latency = interconnect_module.link_module.latency
        flit_size = interconnect_module.link_module.flit_size
        header_size = interconnect_module.link_module.header_size
        max_payload_size = interconnect_module.link_module.max_payload_size
        link_count_per_device = interconnect_module.link_count_per_device
        data_size = size(self.input_shape) * self.data_type.word_size
        if interconnect_module.topology == TopologyType.FC:
            edge_bandwidth_per_direction = (
                link_bandwidth_per_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_bandwidth_both_direction = (
                link_bandwidth_both_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_latency = link_latency
            data_size_per_device = data_size / device_count
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            # stage 1: ring reduce
            latency = (
                edge_latency
                + effective_data_size_per_device / edge_bandwidth_both_direction
            ) * (device_count - 1)
            # stage 2: broadcast
            latency += effective_data_size_per_device / edge_bandwidth_per_direction
            latency += (
                data_size / interconnect_module.internal_link_bandwidth_per_direction
            )
            self.latency = latency
            return latency
        elif interconnect_module.topology == TopologyType.RING:
            edge_bandwidth = link_bandwidth_per_direction * link_count_per_device
            edge_latency = link_latency
            data_size_per_device = data_size / device_count
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            per_transmission_latency = effective_data_size_per_device / edge_bandwidth
            latency = (edge_latency + per_transmission_latency) * (
                (device_count - 1) * 2
            )
            latency += (
                data_size / interconnect_module.internal_link_bandwidth_per_direction
            )
            self.latency = latency
        else:
            raise NotImplementedError
        return self.latency


if __name__ == "__main__":
    reduce = AllReduceMultiPCB(data_type_dict["fp16"])
    tensor = Tensor([8, 2048, 12288], data_type=data_type_dict["fp16"])
    reduce(tensor)
    reduce.simulate(interconnect_module_dict["A100"])

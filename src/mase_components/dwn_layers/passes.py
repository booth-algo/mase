from chop.passes.graph.analysis.add_metadata.add_hardware_metadata import *
from collections import OrderedDict


def dwn_hardware_metadata_pass(graph, args={}):
    """
    Add hardware metadata to DWN LUTLayer nodes for Verilog emit.
    Modelled after difflogic_hardware_metadata_optimize_pass.
    """
    def _is_dwn_lut_node(node):
        return node.meta["mase"]["common"]["mase_op"] == "user_defined_module"

    for node in graph.nodes:
        if _is_dwn_lut_node(node):
            pre_common_args_md = node.meta["mase"]["common"]["args"]
            post_common_args_md = {}
            node.meta["mase"]["hardware"]["dwn_args"] = {}
            for k, v in pre_common_args_md.items():
                if "data_in" not in k:
                    node.meta["mase"]["hardware"]["dwn_args"][k] = v
                else:
                    post_common_args_md[k] = v
            post_common_args_md = OrderedDict(post_common_args_md)
            node.meta["mase"]["common"]["args"] = post_common_args_md
            node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL_RTL"
            node.meta["mase"]["hardware"]["module"] = "fixed_dwn_lut_layer"
            node.meta["mase"]["hardware"]["dependence_files"] = [
                "dwn_layers/rtl/fixed_dwn_lut_neuron.sv",
                "dwn_layers/rtl/fixed_dwn_lut_layer.sv",
            ]
    return (graph, None)


def dwn_hardware_force_fixed_flatten_pass(graph, args={}):
    """
    Force the flatten node (float->binary boundary) to use fixed_dwn_flatten RTL.
    Mirrors difflogic_hardware_force_fixed_flatten_pass.
    """
    for node in graph.nodes:
        if node.meta["mase"]["common"]["mase_op"] == "flatten":
            node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL_RTL"
            node.meta["mase"]["hardware"]["module"] = "fixed_dwn_flatten"
            node.meta["mase"]["hardware"]["dependence_files"] = [
                "dwn_layers/rtl/fixed_dwn_flatten.sv"
            ]
            add_verilog_param(node)
            add_extra_verilog_param(node, graph)
            graph.meta["mase"]["hardware"]["verilog_sources"] += \
                node.meta["mase"]["hardware"]["dependence_files"]
    return (graph, None)


def dwn_hardware_groupsum_pass(graph, args={}):
    """Add hardware metadata to GroupSum nodes."""
    def _is_dwn_groupsum_node(node):
        # GroupSum will appear as a user_defined_module with 'groupsum' in the name
        if node.op != "call_module":
            return False
        module_name = node.target if isinstance(node.target, str) else ""
        return "groupsum" in module_name.lower() or "group_sum" in module_name.lower()

    for node in graph.nodes:
        if _is_dwn_groupsum_node(node):
            node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL_RTL"
            node.meta["mase"]["hardware"]["module"] = "fixed_dwn_groupsum"
            node.meta["mase"]["hardware"]["dependence_files"] = [
                "dwn_layers/rtl/fixed_dwn_groupsum.sv"
            ]
    return (graph, None)

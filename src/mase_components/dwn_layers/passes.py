from chop.passes.graph.analysis.add_metadata.add_hardware_metadata import *
from chop.passes.graph.utils import get_module_by_name
from collections import OrderedDict


def _pack_dwn_lut_params(lut_layer):
    """
    Pack a trained LUTLayer's indices and contents into Verilog hex bit-vector literals.

    Returns (input_size, output_size, lut_n, indices_hex, contents_hex) where
    indices_hex and contents_hex are strings of the form ``"<N>'h<HEX>"`` ready to
    embed directly in a Verilog parameter declaration.
    """
    indices = lut_layer.get_input_indices()   # (output_size, n) int32 tensor
    output_size = lut_layer.output_size
    lut_n = lut_layer.n
    lut_entries = 2 ** lut_n

    # Pack INPUT_INDICES: indices[i,k] → bits [(i*lut_n+k)*8 +: 8]
    packed_indices = 0
    for i in range(output_size):
        for k in range(lut_n):
            idx = int(indices[i, k].item()) & 0xFF
            packed_indices |= idx << ((i * lut_n + k) * 8)

    # Pack LUT_CONTENTS: contents[i,j] → bit [i*lut_entries + j]
    contents = lut_layer.get_lut_contents()   # (output_size, 2^n) int tensor
    packed_contents = 0
    for i in range(output_size):
        for j in range(lut_entries):
            bit = int(contents[i, j].item()) & 1
            packed_contents |= bit << (i * lut_entries + j)

    # Format as Verilog hex literals with explicit bit-width
    indices_bits = output_size * lut_n * 8
    contents_bits = output_size * lut_entries
    indices_hex = f"{indices_bits}'h{packed_indices:0{(indices_bits + 3) // 4}X}"
    contents_hex = f"{contents_bits}'h{packed_contents:0{(contents_bits + 3) // 4}X}"

    return lut_layer.input_size, output_size, lut_n, indices_hex, contents_hex


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
            # Reset interface (remove stale BRAM entries from earlier passes)
            node.meta["mase"]["hardware"]["interface"] = {}

            # Extract DWN-specific parameters from the trained LUTLayer and pack
            # into Verilog hex literals.  These replace the generic DATA_IN/DATA_OUT
            # params that add_verilog_param would otherwise emit, so that the
            # generated dwn_top.sv declares INPUT_SIZE / OUTPUT_SIZE / LUT_N /
            # INPUT_INDICES / LUT_CONTENTS — which is what fixed_dwn_lut_layer.sv
            # actually expects.
            lut_layer = get_module_by_name(graph.model, node.target)
            in_sz, out_sz, lut_n, idx_hex, cont_hex = _pack_dwn_lut_params(lut_layer)
            node.meta["mase"]["hardware"]["verilog_param"] = {
                "INPUT_SIZE": in_sz,
                "OUTPUT_SIZE": out_sz,
                "LUT_N": lut_n,
                "INPUT_INDICES": idx_hex,
                "LUT_CONTENTS": cont_hex,
            }

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
            node.meta["mase"]["hardware"]["interface"] = {}
            add_verilog_param(node)
            add_extra_verilog_param(node, graph)
    return (graph, None)

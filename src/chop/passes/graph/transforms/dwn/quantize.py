import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

DWN_CONVERTIBLE_OPS = ("linear",)


def _get_mase_op(node):
    """Get the mase operation type for a node."""
    try:
        return node.meta["mase"]["common"]["mase_op"]
    except (KeyError, AttributeError):
        return None


def _get_node_actual_target(graph, node):
    """Get the actual nn.Module for a call_module node."""
    if node.op != "call_module":
        return None
    # Walk the module hierarchy to find the submodule
    parts = node.target.split(".")
    module = graph.model
    for part in parts:
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def _set_node_module(graph, node, new_module):
    """Set a submodule in the graph model by node target path."""
    parts = node.target.split(".")
    parent = graph.model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _graph_iterator_dwn_by_type(graph, pass_args: dict):
    """
    Standalone graph iterator that converts nn.Linear nodes to LUTLayer.
    Does NOT use create_new_module() or quantized_module_map.

    lut_n in config may be an int (uniform across all layers) or a list
    (per-layer fan-in, indexed in the order Linear nodes are encountered).
    """
    # Lazy import to avoid circular imports
    from chop.nn.dwn import LUTLayer

    linear_config = pass_args.get("linear", pass_args.get("default", {}))
    node_config = linear_config.get("config", {})

    if node_config.get("name") != "dwn":
        logger.warning("DWN transform pass: no 'dwn' config found, skipping all nodes")
        return graph

    lut_n_cfg = node_config.get("lut_n", 6)
    lut_n_list = lut_n_cfg if isinstance(lut_n_cfg, list) else None
    linear_counter = 0

    for node in graph.fx_graph.nodes:
        if node.op != "call_module":
            continue

        # Check if this is a linear layer
        actual_module = _get_node_actual_target(graph, node)
        if actual_module is None or not isinstance(actual_module, nn.Linear):
            continue

        in_features = actual_module.in_features
        n_for_layer = lut_n_list[linear_counter] if lut_n_list is not None else lut_n_cfg

        new_module = LUTLayer(
            input_size=in_features,
            output_size=node_config.get("hidden_size", 2000),
            n=n_for_layer,
            mapping=node_config.get("mapping_first", "learnable"),
            ste=node_config.get("ste", True),
            clamp_luts=node_config.get("clamp_luts", True),
        )

        _set_node_module(graph, node, new_module)
        logger.info(
            f"DWN: Replaced {node.target} ({in_features} -> {node_config.get('hidden_size', 2000)}, lut_n={n_for_layer})"
        )
        linear_counter += 1

    return graph


def _graph_iterator_dwn_by_name(graph, pass_args: dict):
    """Replace specific named modules with DWN LUTLayer."""
    from chop.nn.dwn import LUTLayer

    for node in graph.fx_graph.nodes:
        if node.op != "call_module":
            continue

        node_name = node.target
        if node_name not in pass_args:
            continue

        node_config = pass_args[node_name].get("config", {})
        if node_config.get("name") != "dwn":
            continue

        actual_module = _get_node_actual_target(graph, node)
        if actual_module is None or not isinstance(actual_module, nn.Linear):
            continue

        new_module = LUTLayer(
            input_size=actual_module.in_features,
            output_size=node_config.get("hidden_size", 2000),
            n=node_config.get("lut_n", 6),
            mapping=node_config.get("mapping_first", "learnable"),
        )
        _set_node_module(graph, node, new_module)

    return graph


def dwn_transform_pass(graph, pass_args: dict = None) -> tuple:
    """
    Convert a standard neural network to a DWN.

    pass_args example:
        {
            "by": "type",
            "default": {"config": {"name": None}},
            "linear": {
                "config": {
                    "name": "dwn",
                    "lut_n": 6,
                    "hidden_size": 2000,
                    "num_layers": 2,
                    "mapping_first": "learnable",
                    "mapping_rest": "random",
                    "num_bits": 8,
                    "tau": 3.33,
                    "lambda_reg": 1e-4,
                }
            },
        }
    """
    if pass_args is None:
        pass_args = {}

    pass_args = dict(pass_args)  # don't mutate caller's dict
    by = pass_args.pop("by", "type")

    match by:
        case "type":
            graph = _graph_iterator_dwn_by_type(graph, pass_args)
        case "name":
            graph = _graph_iterator_dwn_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported DWN "by": {by}')

    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}

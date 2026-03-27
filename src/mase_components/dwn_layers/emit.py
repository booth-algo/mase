"""DWN RTL emission - callable library function.

Emit synthesisable SystemVerilog from a trained DWN checkpoint or a
pre-built DWNHardwareCore model.  Supports LUT-only emission and
full-pipeline (thermometer + LUT + groupsum) via the `full_pipeline` flag.

Can be called programmatically (e.g. from testbenches) without argparse.
"""

import os
import re
import shutil

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Path to mase_components RTL source tree (resolved relative to this file)
# ---------------------------------------------------------------------------
_MASE_COMPONENTS_RTL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)



def _load_model_from_checkpoint(ckpt_path, device):
    """Load a .pt checkpoint, reconstruct DWNModel, return (hw_model, cfg, full_model, state_dict)."""
    from chop.nn.dwn import DWNModel
    from mase_components.dwn_layers.hardware_core import DWNHardwareCore

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("model_config")
    if cfg is None:
        raise ValueError(f"Checkpoint missing 'model_config' key: {ckpt_path}")

    # Strip training-only keys not accepted by DWNModel.__init__
    model_kwargs = {k: v for k, v in cfg.items()
                    if k not in ("area_lambda", "lambda_reg")}
    model = DWNModel(**model_kwargs).to(device)

    # fit_thermometer registers the thresholds buffer so load_state_dict accepts it.
    # The placeholder data is immediately overwritten by checkpoint values.
    model.fit_thermometer(torch.zeros(2, cfg["input_features"]).to(device))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    hw_model = DWNHardwareCore(list(model.lut_layers)).to(device)
    hw_model.eval()

    return hw_model, cfg, model, ckpt["model_state_dict"]


def _run_mase_emit_pipeline(hw_model, dummy_input, output_dir, top_name):
    """Build MaseGraph, run metadata + emit passes. Return (graph, rtl_dir)."""
    from chop.nn.dwn import LUTLayer
    from chop.ir.graph.mase_graph import MaseGraph
    from chop.passes.graph.analysis.init_metadata import init_metadata_analysis_pass
    from chop.passes.graph.analysis.add_metadata.add_common_metadata import (
        add_common_metadata_analysis_pass,
    )
    from chop.passes.graph.analysis.add_metadata.add_hardware_metadata import (
        add_hardware_metadata_analysis_pass,
    )
    from chop.passes.graph.transforms.verilog.emit_top import (
        emit_verilog_top_transform_pass,
    )
    from chop.passes.graph.transforms.verilog.emit_internal import (
        emit_internal_rtl_transform_pass,
    )
    from mase_components.dwn_layers.passes import dwn_hardware_metadata_pass

    custom_ops = {
        "modules": {
            LUTLayer: {
                "args": {"x": "data_in"},
                "module": "fixed_dwn_lut_layer",
                "dependence_files": [
                    "dwn_layers/rtl/fixed/fixed_dwn_lut_neuron.sv",
                    "dwn_layers/rtl/fixed/fixed_dwn_lut_layer.sv",
                ],
            }
        },
        "functions": {},
    }

    graph = MaseGraph(hw_model, custom_ops=custom_ops)
    graph, _ = init_metadata_analysis_pass(graph)
    graph, _ = add_common_metadata_analysis_pass(
        graph, pass_args={"dummy_in": {"x": dummy_input}, "add_value": False},
    )
    graph, _ = add_hardware_metadata_analysis_pass(graph)
    graph, _ = dwn_hardware_metadata_pass(graph)
    graph, _ = emit_verilog_top_transform_pass(
        graph, pass_args={"project_dir": output_dir, "top_name": top_name},
    )
    graph, _ = emit_internal_rtl_transform_pass(
        graph, pass_args={"project_dir": output_dir},
    )

    rtl_dir = os.path.join(output_dir, "hardware", "rtl")
    return graph, rtl_dir


def _emit_pipelined_variant(rtl_dir, top_name):
    """Create dwn_top_clocked.sv via regex substitution + copy clocked layer component."""
    top_sv_path = os.path.join(rtl_dir, f"{top_name}.sv")
    with open(top_sv_path, "r") as f:
        top_sv = f.read()

    clocked_sv = re.sub(rf"\bmodule {re.escape(top_name)}\b",
                         f"module {top_name}_clocked", top_sv)
    clocked_sv = re.sub(r"\bfixed_dwn_lut_layer\b",
                         "fixed_dwn_lut_layer_clocked", clocked_sv)

    clocked_top_path = os.path.join(rtl_dir, f"{top_name}_clocked.sv")
    with open(clocked_top_path, "w") as f:
        f.write(clocked_sv)

    clocked_layer_src = os.path.join(
        _MASE_COMPONENTS_RTL, "dwn_layers/rtl/fixed/fixed_dwn_lut_layer_clocked.sv"
    )
    shutil.copy(clocked_layer_src, os.path.join(rtl_dir, "fixed_dwn_lut_layer_clocked.sv"))

    return clocked_top_path



def _pack_thresholds_sv(thresholds_int, num_features, num_thresholds, feature_width):
    """Pack integer thresholds into a SystemVerilog parameter hex literal."""
    total_bits = num_features * num_thresholds * feature_width
    value = 0
    for f in range(num_features):
        for t in range(num_thresholds):
            thresh = int(thresholds_int[f][t]) & ((1 << feature_width) - 1)
            value |= thresh << ((f * num_thresholds + t) * feature_width)
    hex_digits = (total_bits + 3) // 4
    return f"{total_bits}'h{value:0{hex_digits}X}"


def _emit_wrapper(rtl_dir, cfg, thresholds_sv, feature_width):
    """Emit full_pipeline_top.sv - combinational full pipeline."""
    num_features = cfg["input_features"]
    num_bits = cfg["num_bits"]
    num_classes = cfg["num_classes"]
    lut_output = cfg["hidden_sizes"][-1]
    thermo_bits = num_features * num_bits

    sv = f"""\
// Full DWN inference pipeline: thermometer encoding -> LUT stack -> GroupSum
// Auto-generated by dwn.emit
// Config: input_features={num_features}, num_bits={num_bits}, num_classes={num_classes}
// Input: {num_features} features x {feature_width}-bit integers
// Thermometer output: {thermo_bits} bits -> LUT stack -> {lut_output} bits -> {num_classes} class scores

module full_pipeline_top (
    input  wire  [{num_features * feature_width - 1}:0] data_in_0,
    input  logic                             data_in_0_valid,
    output logic                             data_in_0_ready,

    output logic [({num_classes}*($clog2({lut_output}/{num_classes})+1))-1:0] data_out_0,
    output logic                             data_out_0_valid,
    input  logic                             data_out_0_ready,

    input  wire                              clk,
    input  wire                              rst
);

    // ----------------------------------------------------------------
    // Stage 1: Thermometer encoding
    // ----------------------------------------------------------------
    wire  [{thermo_bits - 1}:0] thermo_out;
    logic                       thermo_valid;
    logic                       thermo_ready;

    fixed_dwn_thermometer #(
        .NUM_FEATURES   ({num_features}),
        .FEATURE_WIDTH  ({feature_width}),
        .NUM_THRESHOLDS ({num_bits}),
        .THRESHOLDS     ({thresholds_sv})
    ) thermo_inst (
        .data_in_0        (data_in_0),
        .data_in_0_valid  (data_in_0_valid),
        .data_in_0_ready  (data_in_0_ready),
        .data_out_0       (thermo_out),
        .data_out_0_valid (thermo_valid),
        .data_out_0_ready (thermo_ready)
    );

    // ----------------------------------------------------------------
    // Stage 2: LUT layer stack (generated dwn_top)
    // ----------------------------------------------------------------
    wire  [{lut_output - 1}:0] lut_out;
    logic                      lut_valid;
    logic                      lut_ready;

    dwn_top lut_inst (
        .clk              (clk),
        .rst              (rst),
        .data_in_0        (thermo_out),
        .data_in_0_valid  (thermo_valid),
        .data_in_0_ready  (thermo_ready),
        .data_out_0       (lut_out),
        .data_out_0_valid (lut_valid),
        .data_out_0_ready (lut_ready)
    );

    // ----------------------------------------------------------------
    // Stage 3: GroupSum output aggregation
    // ----------------------------------------------------------------
    logic [$clog2({lut_output}/{num_classes}):0] groupsum_packed [0:{num_classes - 1}];

    fixed_dwn_groupsum #(
        .INPUT_SIZE ({lut_output}),
        .NUM_GROUPS ({num_classes})
    ) gs_inst (
        .data_in_0        (lut_out),
        .data_in_0_valid  (lut_valid),
        .data_in_0_ready  (lut_ready),
        .data_out_0       (groupsum_packed),
        .data_out_0_valid (data_out_0_valid),
        .data_out_0_ready (data_out_0_ready)
    );

    assign data_out_0 = {{{','.join(f' groupsum_packed[{i}]' for i in range(num_classes))}}};

endmodule
"""
    out_path = os.path.join(rtl_dir, "full_pipeline_top.sv")
    with open(out_path, "w") as f:
        f.write(sv)
    return out_path


def _emit_clocked_wrapper(rtl_dir, cfg, thresholds_sv, feature_width):
    """Emit full_pipeline_top_clocked.sv with FF register stages."""
    num_features = cfg["input_features"]
    num_bits = cfg["num_bits"]
    num_classes = cfg["num_classes"]
    lut_output = cfg["hidden_sizes"][-1]
    thermo_bits = num_features * num_bits
    score_width = f"$clog2({lut_output}/{num_classes})"

    sv = f"""\
// Full DWN inference pipeline - clocked/registered variant.
// Stages: thermometer (comb) -> FF reg -> dwn_top_clocked -> groupsum (comb) -> FF reg
// Auto-generated by dwn.emit
// Config: input_features={num_features}, num_bits={num_bits}, num_classes={num_classes}
// Input : {num_features} features x {feature_width}-bit integers
// Thermo: {thermo_bits} bits  LUT stack: {lut_output} bits  Output: {num_classes} class scores

module full_pipeline_top_clocked (
    input  wire                                clk,
    input  wire                                rst,
    input  wire  [{num_features * feature_width - 1}:0] data_in_0,
    output logic [{score_width}:0]    data_out_0 [0:{num_classes - 1}]
);

    // ----------------------------------------------------------------
    // Stage 1: Thermometer encoding (combinational)
    // ----------------------------------------------------------------
    wire [{thermo_bits - 1}:0] thermo_comb;

    fixed_dwn_thermometer #(
        .NUM_FEATURES   ({num_features}),
        .FEATURE_WIDTH  ({feature_width}),
        .NUM_THRESHOLDS ({num_bits}),
        .THRESHOLDS     ({thresholds_sv})
    ) thermo_inst (
        .data_in_0        (data_in_0),
        .data_in_0_valid  (1'b1),
        .data_in_0_ready  (),
        .data_out_0       (thermo_comb),
        .data_out_0_valid (),
        .data_out_0_ready (1'b1)
    );

    // ----------------------------------------------------------------
    // Stage 2: Register thermometer output (FF boundary)
    // ----------------------------------------------------------------
    logic [{thermo_bits - 1}:0] thermo_reg;
    always_ff @(posedge clk) begin
        if (rst) thermo_reg <= '0;
        else     thermo_reg <= thermo_comb;
    end

    // ----------------------------------------------------------------
    // Stage 3: LUT layer stack - clocked variant (FFs between layers)
    // ----------------------------------------------------------------
    wire [{lut_output - 1}:0] lut_out;

    dwn_top_clocked lut_inst (
        .clk              (clk),
        .rst              (rst),
        .data_in_0        (thermo_reg),
        .data_in_0_valid  (1'b1),
        .data_in_0_ready  (),
        .data_out_0       (lut_out),
        .data_out_0_valid (),
        .data_out_0_ready (1'b1)
    );

    // ----------------------------------------------------------------
    // Stage 4: GroupSum (combinational)
    // ----------------------------------------------------------------
    wire [{score_width}:0] gs_comb [0:{num_classes - 1}];

    fixed_dwn_groupsum #(
        .INPUT_SIZE ({lut_output}),
        .NUM_GROUPS ({num_classes})
    ) gs_inst (
        .data_in_0        (lut_out),
        .data_in_0_valid  (1'b1),
        .data_in_0_ready  (),
        .data_out_0       (gs_comb),
        .data_out_0_valid (),
        .data_out_0_ready (1'b1)
    );

    // ----------------------------------------------------------------
    // Stage 5: Register outputs (FF boundary)
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < {num_classes}; i++) data_out_0[i] <= '0;
        end else begin
            for (int i = 0; i < {num_classes}; i++) data_out_0[i] <= gs_comb[i];
        end
    end

endmodule
"""
    out_path = os.path.join(rtl_dir, "full_pipeline_top_clocked.sv")
    with open(out_path, "w") as f:
        f.write(sv)
    return out_path


def _emit_paper_scope_wrapper(rtl_dir, cfg):
    """Emit dwn_top_paper_scope.sv: LUT layers + pipelined GroupSum, no thermometer."""
    num_classes = cfg["num_classes"]
    lut_output = cfg["hidden_sizes"][-1]
    thermo_bits = cfg["input_features"] * cfg["num_bits"]
    score_width = f"$clog2({lut_output}/{num_classes})"

    sv = f"""\
// DWN paper-scope top: LUT layers + pipelined GroupSum (no thermometer encoder)
// Matches Table 2 OOC synthesis scope from Bacellar et al. (arXiv:2410.11112)
// Auto-generated by dwn.emit
// Input : {thermo_bits}-bit thermometer-encoded binary
// Output: {num_classes} class scores
module dwn_top_paper_scope (
    input  wire           clk,
    input  wire           rst,
    input  wire  [{thermo_bits - 1}:0] data_in_0,
    output logic [{score_width}:0] data_out_0 [0:{num_classes - 1}]
);

    wire [{lut_output - 1}:0] lut_out;
    wire                      lut_valid;

    dwn_top_clocked lut_inst (
        .clk              (clk),
        .rst              (rst),
        .data_in_0        (data_in_0),
        .data_in_0_valid  (1'b1),
        .data_in_0_ready  (),
        .data_out_0       (lut_out),
        .data_out_0_valid (lut_valid),
        .data_out_0_ready (1'b1)
    );

    fixed_dwn_groupsum_pipelined #(
        .INPUT_SIZE ({lut_output}),
        .NUM_GROUPS ({num_classes})
    ) gs_inst (
        .clk              (clk),
        .rst              (rst),
        .data_in_0        (lut_out),
        .data_in_0_valid  (lut_valid),
        .data_in_0_ready  (),
        .data_out_0       (data_out_0),
        .data_out_0_valid (),
        .data_out_0_ready (1'b1)
    );

endmodule
"""
    out_path = os.path.join(rtl_dir, "dwn_top_paper_scope.sv")
    with open(out_path, "w") as f:
        f.write(sv)
    return out_path


def _emit_full_pipeline(rtl_dir, cfg, state_dict, feature_width):
    """Emit full-pipeline wrappers (thermometer + LUT + groupsum) and copy components."""
    # Extract and quantize thresholds
    thres_key = "thermometer.thresholds"
    num_features = cfg["input_features"]
    num_bits = cfg["num_bits"]
    maxval = (1 << feature_width) - 1

    if state_dict is not None and thres_key in state_dict:
        thresholds_float = state_dict[thres_key]
        thresholds_int = (thresholds_float.clamp(0.0, 1.0) * maxval).round().long()
    else:
        step = maxval // (num_bits + 1)
        thresholds_int = torch.tensor(
            [[step * (t + 1) for t in range(num_bits)] for _ in range(num_features)],
            dtype=torch.long,
        )

    thresholds_sv = _pack_thresholds_sv(thresholds_int, num_features, num_bits, feature_width)

    # Copy component RTL files
    component_src = os.path.join(_MASE_COMPONENTS_RTL, "dwn_layers/rtl/fixed")
    for fname in [
        "fixed_dwn_thermometer.sv",
        "fixed_dwn_groupsum.sv",
        "fixed_dwn_lut_layer_clocked.sv",
        "fixed_dwn_groupsum_pipelined.sv",
    ]:
        src = os.path.join(component_src, fname)
        dst = os.path.join(rtl_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    _emit_wrapper(rtl_dir, cfg, thresholds_sv, feature_width)
    _emit_clocked_wrapper(rtl_dir, cfg, thresholds_sv, feature_width)
    _emit_paper_scope_wrapper(rtl_dir, cfg)


def emit_dwn_rtl(
    *,
    ckpt_path=None,
    model=None,
    model_config=None,
    output_dir,
    top_name="dwn_top",
    full_pipeline=False,
    emit_pipelined=False,
    emit_blif=False,
    feature_width=8,
    device="auto",
):
    """Emit DWN LUT-layer RTL via MASE pipeline.

    Provide EITHER ``ckpt_path`` OR (``model`` + ``model_config``).

    Parameters
    ----------
    ckpt_path : str, optional
        Path to a .pt checkpoint produced by DWN training.
    model : nn.Module, optional
        Pre-built DWNHardwareCore (or compatible nn.Module with .lut_layers).
    model_config : dict, optional
        Required when ``model`` is provided.  Must contain at minimum
        ``input_features`` and ``num_bits`` (or ``thermo_width``).
    output_dir : str
        Root output directory.  RTL is written to ``output_dir/hardware/rtl/``.
    top_name : str
        Top-level Verilog module name (default: ``dwn_top``).
    full_pipeline : bool
        Also emit thermometer + groupsum wrappers (full_pipeline_top*.sv).
    emit_pipelined : bool
        Also emit a clocked/pipelined variant (dwn_top_clocked.sv).
    emit_blif : bool
        Also emit a BLIF file for ABC Boolean minimisation.
    feature_width : int
        Bits per input feature for full-pipeline thermometer (default: 8).
    device : str
        Device for model evaluation ("auto", "cuda", "cpu").

    Returns
    -------
    dict with keys: output_dir, rtl_dir, sv_files, graph.
    """
    # --- input validation ---
    if ckpt_path is not None and model is not None:
        raise ValueError("Provide ckpt_path OR model, not both")
    if ckpt_path is None and model is None:
        raise ValueError("Must provide either ckpt_path or model")
    if model is not None and model_config is None:
        raise ValueError("model_config is required when providing model directly")

    # --- resolve device ---
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # --- get hw_model, config, and optionally full_model + state_dict ---
    full_model = None
    state_dict = None

    if ckpt_path is not None:
        hw_model, cfg, full_model, state_dict = _load_model_from_checkpoint(ckpt_path, dev)
    else:
        hw_model = model.to(dev)
        hw_model.eval()
        cfg = model_config

    # --- compute dummy input shape ---
    thermo_width = cfg.get("thermo_width") or cfg["input_features"] * cfg["num_bits"]
    dummy_input = torch.zeros(1, thermo_width).to(dev)

    # --- run MASE emit pipeline ---
    graph, rtl_dir = _run_mase_emit_pipeline(hw_model, dummy_input, output_dir, top_name)

    # --- optional: pipelined variant ---
    if emit_pipelined:
        _emit_pipelined_variant(rtl_dir, top_name)

    # --- optional: full pipeline wrappers ---
    if full_pipeline:
        if ckpt_path is None and state_dict is None:
            raise ValueError(
                "full_pipeline=True requires ckpt_path (needs checkpoint state_dict for thresholds)"
            )
        # full_pipeline implies pipelined (needs dwn_top_clocked.sv)
        if not emit_pipelined:
            _emit_pipelined_variant(rtl_dir, top_name)
        _emit_full_pipeline(rtl_dir, cfg, state_dict, feature_width)

    # --- optional: BLIF ---
    if emit_blif:
        if full_model is None:
            raise ValueError("emit_blif requires ckpt_path (needs full DWNModel)")
        from mase_components.dwn_layers.blif import emit_network_blif
        blif_path = os.path.join(output_dir, "network.blif")
        emit_network_blif(full_model, blif_path)

    # --- collect results ---
    sv_files = sorted(f for f in os.listdir(rtl_dir) if f.endswith(".sv"))

    return {
        "output_dir": os.path.abspath(output_dir),
        "rtl_dir": os.path.abspath(rtl_dir),
        "sv_files": sv_files,
        "graph": graph,
    }

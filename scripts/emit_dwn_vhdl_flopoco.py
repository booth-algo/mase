#!/usr/bin/env python3
"""
Emit FloPoCo-style VHDL from a trained DWN checkpoint.

Each neuron is a SEPARATE entity with embedded truth table (with...select),
preventing Vivado from seeing across entity boundaries for WAFR packing analysis.

Usage:
    python scripts/emit_dwn_vhdl_flopoco.py --ckpt-name baseline_n6
    python scripts/emit_dwn_vhdl_flopoco.py --ckpt /path/to/checkpoint.pt

Output directory: mase_output/dwn/{ckpt_name}_vhdl/
"""
import argparse
import math
import os
import sys

# ---------------------------------------------------------------------------
# sys.modules stubs to bypass heavy MASE imports (same pattern as emit_dwn_rtl.py)
# ---------------------------------------------------------------------------
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, _src)

# sys.modules stubs to bypass heavy MASE imports (chop.__init__ pulls MaseGraph
# which pulls transformers which can fail).  Same pattern as run_dwn_training.py.
import types
for _pkg in ['chop', 'chop.nn']:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_src, *_pkg.split('.'))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt-name", type=str, default="baseline_n6",
                    help="Checkpoint stem (looks in mase_output/dwn/<name>.pt)")
    p.add_argument("--ckpt", type=str, default=None,
                    help="Full checkpoint path (overrides --ckpt-name)")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: mase_output/dwn/<ckpt-name>_vhdl)")
    p.add_argument("--input-bits", type=int, default=8,
                    help="Bit width of raw input features (for threshold scaling)")
    p.add_argument("--pipelined", action="store_true",
                    help="Generate registered pipeline stages in dwn_top (clk/rst ports); "
                         "sub-entities remain combinational")
    return p.parse_args()


# ---------------------------------------------------------------------------
# VHDL generation helpers
# ---------------------------------------------------------------------------

def vhdl_header():
    return ("library ieee;\n"
            "use ieee.std_logic_1164.all;\n"
            "use ieee.numeric_std.all;\n\n")


def emit_thermometer_encoder(f, thresholds_tensor, num_bits, input_bits):
    """
    Generate thermometer_encoder.vhd.

    Each feature-threshold pair becomes a comparator instance inside a
    structural architecture.  Thresholds are stored as generics.

    Args:
        f:                  Open file handle (streaming write).
        thresholds_tensor:  Shape (num_features, num_bits), values in [0,1].
        num_bits:           Number of thresholds per feature.
        input_bits:         Bit width of each raw input (e.g. 8 for MNIST).
    """
    num_features = thresholds_tensor.shape[0]
    total_out = num_features * num_bits
    max_val = (1 << input_bits) - 1  # 255 for 8-bit

    # --- Comparator entity (reusable) ---
    f.write(vhdl_header())
    f.write("-- Single threshold comparator: output '1' when input > THRESHOLD\n")
    f.write("-- Implements strict > as >= (THRESHOLD + 1)\n")
    f.write("entity thermo_comparator is\n")
    f.write("  generic (\n")
    f.write(f"    INPUT_BITS : positive := {input_bits};\n")
    f.write("    THRESHOLD  : natural  := 0\n")
    f.write("  );\n")
    f.write("  port (\n")
    f.write(f"    x   : in  unsigned(INPUT_BITS-1 downto 0);\n")
    f.write("    y   : out std_logic\n")
    f.write("  );\n")
    f.write("end entity thermo_comparator;\n\n")
    f.write("architecture rtl of thermo_comparator is\n")
    f.write("begin\n")
    f.write("  -- strict greater-than: x > THRESHOLD  <==>  x >= THRESHOLD+1\n")
    f.write("  y <= '1' when x >= to_unsigned(THRESHOLD + 1, INPUT_BITS) else '0';\n")
    f.write("end architecture rtl;\n\n")

    # --- Top-level thermometer encoder (structural) ---
    f.write("-- " + "=" * 70 + "\n")
    f.write(f"-- Thermometer encoder: {num_features} features x {num_bits} thresholds "
            f"= {total_out} output bits\n")
    f.write("-- " + "=" * 70 + "\n\n")
    f.write(vhdl_header())
    f.write("entity thermometer_encoder is\n")
    f.write("  port (\n")
    f.write(f"    x_in  : in  std_logic_vector({num_features * input_bits - 1} downto 0);\n")
    f.write(f"    x_out : out std_logic_vector({total_out - 1} downto 0)\n")
    f.write("  );\n")
    f.write("end entity thermometer_encoder;\n\n")
    f.write("architecture structural of thermometer_encoder is\n")
    f.write("begin\n")

    for feat_i in range(num_features):
        for bit_j in range(num_bits):
            # Scale normalized threshold [0,1] to [0, max_val] integer
            raw_thresh = float(thresholds_tensor[feat_i, bit_j])
            thresh_int = int(round(raw_thresh * max_val))
            thresh_int = max(0, min(max_val, thresh_int))

            inst_name = f"cmp_f{feat_i}_b{bit_j}"
            out_idx = feat_i * num_bits + bit_j
            in_hi = (feat_i + 1) * input_bits - 1
            in_lo = feat_i * input_bits

            f.write(f"  {inst_name}: entity work.thermo_comparator\n")
            f.write(f"    generic map (INPUT_BITS => {input_bits}, THRESHOLD => {thresh_int})\n")
            f.write(f"    port map (\n")
            f.write(f"      x => unsigned(x_in({in_hi} downto {in_lo})),\n")
            f.write(f"      y => x_out({out_idx})\n")
            f.write(f"    );\n")

    f.write("end architecture structural;\n")


def emit_lut_layer(f, layer_idx, lut_contents, input_indices, input_width):
    """
    Generate lut_layer_{layer_idx}.vhd.

    Each neuron is a SEPARATE entity with its truth table embedded as a
    'with...select' case statement.  A structural top instantiates all neurons.

    Args:
        f:              Open file handle.
        layer_idx:      Layer index (0, 1, ...).
        lut_contents:   Tensor shape (output_size, 2^n), int {0,1}.
        input_indices:  Tensor shape (output_size, n), int32.
        input_width:    Total width of this layer's input vector.
    """
    output_size = lut_contents.shape[0]
    num_entries = lut_contents.shape[1]
    n = int(math.log2(num_entries))

    # --- Individual neuron entities ---
    for neuron_i in range(output_size):
        entity_name = f"lut_L{layer_idx}_N{neuron_i}"
        lut_row = lut_contents[neuron_i]

        f.write(vhdl_header())
        f.write(f"entity {entity_name} is\n")
        f.write(f"  port (\n")
        f.write(f"    lut_in  : in  std_logic_vector({n - 1} downto 0);\n")
        f.write(f"    lut_out : out std_logic\n")
        f.write(f"  );\n")
        f.write(f"end entity {entity_name};\n\n")

        f.write(f"architecture rtl of {entity_name} is\n")
        f.write(f"begin\n")
        f.write(f"  with lut_in select lut_out <=\n")

        # Enumerate all 2^n entries; MSB-first bit ordering
        for j in range(num_entries):
            val = "'1'" if int(lut_row[j]) == 1 else "'0'"
            bits = format(j, f"0{n}b")  # MSB first
            # Convert binary string to VHDL std_logic_vector literal
            vhdl_bits = '"' + bits + '"'
            f.write(f"    {val} when {vhdl_bits},\n")

        # Default for non-binary std_logic values (X, U, etc.)
        f.write(f"    '0' when others;\n")

        f.write(f"end architecture rtl;\n\n")

    # --- Structural top for this layer ---
    f.write("-- " + "=" * 70 + "\n")
    f.write(f"-- LUT layer {layer_idx}: {output_size} neurons, fan-in={n}\n")
    f.write("-- " + "=" * 70 + "\n\n")
    f.write(vhdl_header())

    layer_entity = f"lut_layer_{layer_idx}"
    f.write(f"entity {layer_entity} is\n")
    f.write(f"  port (\n")
    f.write(f"    layer_in  : in  std_logic_vector({input_width - 1} downto 0);\n")
    f.write(f"    layer_out : out std_logic_vector({output_size - 1} downto 0)\n")
    f.write(f"  );\n")
    f.write(f"end entity {layer_entity};\n\n")

    f.write(f"architecture structural of {layer_entity} is\n")

    # Declare signal for each neuron's fan-in wiring
    for neuron_i in range(output_size):
        f.write(f"  signal s_in_{neuron_i} : std_logic_vector({n - 1} downto 0);\n")

    f.write(f"begin\n\n")

    for neuron_i in range(output_size):
        indices = input_indices[neuron_i]  # shape (n,)

        # Wire the fan-in: MSB = index 0, LSB = index n-1
        for k in range(n):
            idx = int(indices[k])
            f.write(f"  s_in_{neuron_i}({n - 1 - k}) <= layer_in({idx});\n")

        entity_name = f"lut_L{layer_idx}_N{neuron_i}"
        f.write(f"  inst_{neuron_i}: entity work.{entity_name}\n")
        f.write(f"    port map (lut_in => s_in_{neuron_i}, lut_out => layer_out({neuron_i}));\n\n")

    f.write(f"end architecture structural;\n")


def emit_groupsum(f, num_classes, neurons_per_class):
    """
    Generate groupsum.vhd with explicit binary adder trees.

    Each class group sums neurons_per_class single-bit values into
    a ceil(log2(neurons_per_class+1))-bit result.

    Args:
        f:                  Open file handle.
        num_classes:        Number of output classes.
        neurons_per_class:  Number of bits per group.
    """
    total_in = num_classes * neurons_per_class
    sum_bits = int(math.ceil(math.log2(neurons_per_class + 1)))

    f.write(vhdl_header())
    f.write(f"-- GroupSum: {num_classes} groups x {neurons_per_class} bits, "
            f"{sum_bits}-bit sums\n")
    f.write(f"-- Explicit binary adder tree (NOT popcount/$countones)\n\n")

    # --- Single-group adder tree entity ---
    f.write(vhdl_header())
    f.write(f"entity group_adder_tree is\n")
    f.write(f"  generic (\n")
    f.write(f"    N_INPUTS : positive := {neurons_per_class}\n")
    f.write(f"  );\n")
    f.write(f"  port (\n")
    f.write(f"    bits_in : in  std_logic_vector(N_INPUTS-1 downto 0);\n")
    f.write(f"    sum_out : out unsigned({sum_bits - 1} downto 0)\n")
    f.write(f"  );\n")
    f.write(f"end entity group_adder_tree;\n\n")

    f.write(f"architecture rtl of group_adder_tree is\n")
    f.write(f"begin\n")
    f.write(f"  process(bits_in)\n")
    f.write(f"    variable acc : unsigned({sum_bits - 1} downto 0);\n")
    f.write(f"  begin\n")
    f.write(f"    acc := (others => '0');\n")
    f.write(f"    for i in 0 to N_INPUTS-1 loop\n")
    f.write(f"      if bits_in(i) = '1' then\n")
    f.write(f"        acc := acc + 1;\n")
    f.write(f"      end if;\n")
    f.write(f"    end loop;\n")
    f.write(f"    sum_out <= acc;\n")
    f.write(f"  end process;\n")
    f.write(f"end architecture rtl;\n\n")

    # --- Top-level groupsum ---
    f.write("-- " + "=" * 70 + "\n")
    f.write(f"-- GroupSum top: {num_classes} classes\n")
    f.write("-- " + "=" * 70 + "\n\n")
    f.write(vhdl_header())
    f.write(f"entity groupsum is\n")
    f.write(f"  port (\n")
    f.write(f"    gs_in  : in  std_logic_vector({total_in - 1} downto 0);\n")

    # Output: concatenated sum vectors
    total_out_bits = num_classes * sum_bits
    f.write(f"    gs_out : out std_logic_vector({total_out_bits - 1} downto 0)\n")
    f.write(f"  );\n")
    f.write(f"end entity groupsum;\n\n")

    f.write(f"architecture structural of groupsum is\n")
    for g in range(num_classes):
        f.write(f"  signal grp_{g}_sum : unsigned({sum_bits - 1} downto 0);\n")
    f.write(f"begin\n\n")

    for g in range(num_classes):
        in_lo = g * neurons_per_class
        in_hi = (g + 1) * neurons_per_class - 1
        out_lo = g * sum_bits
        out_hi = (g + 1) * sum_bits - 1

        f.write(f"  grp_{g}: entity work.group_adder_tree\n")
        f.write(f"    generic map (N_INPUTS => {neurons_per_class})\n")
        f.write(f"    port map (\n")
        f.write(f"      bits_in => gs_in({in_hi} downto {in_lo}),\n")
        f.write(f"      sum_out => grp_{g}_sum\n")
        f.write(f"    );\n")
        f.write(f"  gs_out({out_hi} downto {out_lo}) <= std_logic_vector(grp_{g}_sum);\n\n")

    f.write(f"end architecture structural;\n")


def emit_dwn_top(f, num_features, num_bits, input_bits, hidden_sizes, num_classes):
    """
    Generate dwn_top.vhd connecting all blocks.
    """
    thermo_in_width = num_features * input_bits
    thermo_out_width = num_features * num_bits
    last_layer_size = hidden_sizes[-1]
    neurons_per_class = last_layer_size // num_classes
    sum_bits = int(math.ceil(math.log2(neurons_per_class + 1)))
    total_out_bits = num_classes * sum_bits

    f.write(vhdl_header())
    f.write("-- " + "=" * 70 + "\n")
    f.write("-- DWN top-level: thermometer -> LUT layers -> GroupSum\n")
    f.write("-- " + "=" * 70 + "\n\n")
    f.write(f"entity dwn_top is\n")
    f.write(f"  port (\n")
    f.write(f"    x_in  : in  std_logic_vector({thermo_in_width - 1} downto 0);\n")
    f.write(f"    y_out : out std_logic_vector({total_out_bits - 1} downto 0)\n")
    f.write(f"  );\n")
    f.write(f"end entity dwn_top;\n\n")

    f.write(f"architecture structural of dwn_top is\n")

    # Internal signals
    f.write(f"  signal thermo_out : std_logic_vector({thermo_out_width - 1} downto 0);\n")
    for i, sz in enumerate(hidden_sizes):
        f.write(f"  signal layer_{i}_out : std_logic_vector({sz - 1} downto 0);\n")
    f.write(f"begin\n\n")

    # Thermometer encoder
    f.write(f"  thermo_inst: entity work.thermometer_encoder\n")
    f.write(f"    port map (x_in => x_in, x_out => thermo_out);\n\n")

    # LUT layers
    for i, sz in enumerate(hidden_sizes):
        if i == 0:
            in_sig = "thermo_out"
        else:
            in_sig = f"layer_{i - 1}_out"
        f.write(f"  lut_layer_{i}_inst: entity work.lut_layer_{i}\n")
        f.write(f"    port map (layer_in => {in_sig}, layer_out => layer_{i}_out);\n\n")

    # GroupSum
    last_sig = f"layer_{len(hidden_sizes) - 1}_out"
    f.write(f"  gs_inst: entity work.groupsum\n")
    f.write(f"    port map (gs_in => {last_sig}, gs_out => y_out);\n\n")

    f.write(f"end architecture structural;\n")


def emit_dwn_top_pipelined(f, num_features, num_bits, input_bits, hidden_sizes, num_classes):
    """
    Generate dwn_top.vhd with registered pipeline stages between each block.

    Pipeline structure (4 FF stages):
        x_in -> thermometer_encoder -> reg0 -> lut_layer_0 -> reg1
             -> lut_layer_1 -> ... -> reg(N-1) -> groupsum -> reg_out -> y_out

    Sub-entities (thermometer, lut layers, groupsum) remain purely combinational.
    Only dwn_top adds clk/rst ports and inserts the registers.
    """
    thermo_in_width = num_features * input_bits
    thermo_out_width = num_features * num_bits
    last_layer_size = hidden_sizes[-1]
    neurons_per_class = last_layer_size // num_classes
    sum_bits = int(math.ceil(math.log2(neurons_per_class + 1)))
    total_out_bits = num_classes * sum_bits
    num_layers = len(hidden_sizes)

    f.write(vhdl_header())
    f.write("-- " + "=" * 70 + "\n")
    f.write("-- DWN top-level (pipelined): thermometer -> reg -> LUT layers -> reg -> GroupSum -> reg\n")
    f.write("-- Sub-entities are purely combinational; registers live here only.\n")
    f.write("-- " + "=" * 70 + "\n\n")

    f.write(f"entity dwn_top is\n")
    f.write(f"  port (\n")
    f.write(f"    clk   : in  std_logic;\n")
    f.write(f"    rst   : in  std_logic;\n")
    f.write(f"    x_in  : in  std_logic_vector({thermo_in_width - 1} downto 0);\n")
    f.write(f"    y_out : out std_logic_vector({total_out_bits - 1} downto 0)\n")
    f.write(f"  );\n")
    f.write(f"end entity dwn_top;\n\n")

    f.write(f"architecture structural of dwn_top is\n\n")

    # Combinational wires (outputs of each sub-entity)
    f.write(f"  -- Combinational wires (sub-entity outputs)\n")
    f.write(f"  signal thermo_comb  : std_logic_vector({thermo_out_width - 1} downto 0);\n")
    for i, sz in enumerate(hidden_sizes):
        f.write(f"  signal layer_{i}_comb : std_logic_vector({sz - 1} downto 0);\n")
    f.write(f"  signal gs_comb      : std_logic_vector({total_out_bits - 1} downto 0);\n\n")

    # Pipeline registers (one per stage boundary)
    f.write(f"  -- Pipeline registers\n")
    f.write(f"  signal thermo_reg   : std_logic_vector({thermo_out_width - 1} downto 0);\n")
    for i, sz in enumerate(hidden_sizes):
        f.write(f"  signal layer_{i}_reg  : std_logic_vector({sz - 1} downto 0);\n")
    f.write(f"  signal gs_reg       : std_logic_vector({total_out_bits - 1} downto 0);\n\n")

    f.write(f"begin\n\n")

    # Thermometer encoder (combinational)
    f.write(f"  -- Stage 0: thermometer encoder (combinational)\n")
    f.write(f"  thermo_inst: entity work.thermometer_encoder\n")
    f.write(f"    port map (x_in => x_in, x_out => thermo_comb);\n\n")

    # Register: thermo_comb -> thermo_reg
    f.write(f"  -- Register: thermometer output -> lut_layer_0 input\n")
    f.write(f"  process(clk)\n")
    f.write(f"  begin\n")
    f.write(f"    if rising_edge(clk) then\n")
    f.write(f"      if rst = '1' then\n")
    f.write(f"        thermo_reg <= (others => '0');\n")
    f.write(f"      else\n")
    f.write(f"        thermo_reg <= thermo_comb;\n")
    f.write(f"      end if;\n")
    f.write(f"    end if;\n")
    f.write(f"  end process;\n\n")

    # LUT layers with inter-layer registers
    for i, sz in enumerate(hidden_sizes):
        in_reg = "thermo_reg" if i == 0 else f"layer_{i - 1}_reg"
        f.write(f"  -- Stage {i + 1}: lut_layer_{i} (combinational)\n")
        f.write(f"  lut_layer_{i}_inst: entity work.lut_layer_{i}\n")
        f.write(f"    port map (layer_in => {in_reg}, layer_out => layer_{i}_comb);\n\n")

        f.write(f"  -- Register: lut_layer_{i} output -> {'lut_layer_' + str(i+1) + ' input' if i < num_layers - 1 else 'groupsum input'}\n")
        f.write(f"  process(clk)\n")
        f.write(f"  begin\n")
        f.write(f"    if rising_edge(clk) then\n")
        f.write(f"      if rst = '1' then\n")
        f.write(f"        layer_{i}_reg <= (others => '0');\n")
        f.write(f"      else\n")
        f.write(f"        layer_{i}_reg <= layer_{i}_comb;\n")
        f.write(f"      end if;\n")
        f.write(f"    end if;\n")
        f.write(f"  end process;\n\n")

    # GroupSum (combinational)
    last_reg = f"layer_{num_layers - 1}_reg"
    f.write(f"  -- Stage {num_layers + 1}: groupsum (combinational)\n")
    f.write(f"  gs_inst: entity work.groupsum\n")
    f.write(f"    port map (gs_in => {last_reg}, gs_out => gs_comb);\n\n")

    # Output register: gs_comb -> gs_reg -> y_out
    f.write(f"  -- Register: groupsum output -> final output\n")
    f.write(f"  process(clk)\n")
    f.write(f"  begin\n")
    f.write(f"    if rising_edge(clk) then\n")
    f.write(f"      if rst = '1' then\n")
    f.write(f"        gs_reg <= (others => '0');\n")
    f.write(f"      else\n")
    f.write(f"        gs_reg <= gs_comb;\n")
    f.write(f"      end if;\n")
    f.write(f"    end if;\n")
    f.write(f"  end process;\n\n")

    f.write(f"  y_out <= gs_reg;\n\n")
    f.write(f"end architecture structural;\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    ckpt_dir = os.path.join(os.path.dirname(__file__), "../mase_output/dwn")
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")
    ckpt_name = args.ckpt_name if not args.ckpt else os.path.splitext(os.path.basename(args.ckpt))[0]
    suffix = "_vhdl_pipelined" if args.pipelined else "_vhdl"
    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{ckpt_name}{suffix}")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_config"]
    state = ckpt["model_state_dict"]
    print(f"  Config: hidden_sizes={cfg['hidden_sizes']}, lut_n={cfg['lut_n']}, "
          f"num_bits={cfg['num_bits']}, mapping_first={cfg['mapping_first']}")

    input_features = cfg["input_features"]
    num_classes = cfg["num_classes"]
    num_bits = cfg["num_bits"]
    hidden_sizes = cfg["hidden_sizes"]
    input_bits = args.input_bits

    # --- Reconstruct model to get LUT contents and input indices ---
    from chop.nn.dwn import DWNModel
    model_kwargs = {k: v for k, v in cfg.items()
                    if k not in ("area_lambda", "lambda_reg")}
    model = DWNModel(**model_kwargs)
    model.fit_thermometer(torch.zeros(2, input_features))
    model.load_state_dict(state)
    model.eval()

    # Extract thresholds
    thresholds = model.thermometer.thresholds  # (num_features, num_bits)
    print(f"  Thresholds: shape={thresholds.shape}, "
          f"range=[{thresholds.min():.4f}, {thresholds.max():.4f}]")

    # Extract per-layer data
    lut_layers = list(model.lut_layers)
    num_layers = len(lut_layers)

    # --- 1. Thermometer encoder ---
    thermo_path = os.path.join(output_dir, "thermometer_encoder.vhd")
    print(f"  Writing {thermo_path} ...")
    with open(thermo_path, "w") as f:
        emit_thermometer_encoder(f, thresholds, num_bits, input_bits)
    print(f"    {input_features} features x {num_bits} thresholds "
          f"= {input_features * num_bits} comparators")

    # --- 2. LUT layers ---
    layer_input_width = input_features * num_bits  # first layer input
    for li, layer in enumerate(lut_layers):
        lut_contents = layer.get_lut_contents().cpu()    # (output_size, 2^n)
        input_indices = layer.get_input_indices().cpu()   # (output_size, n)

        layer_path = os.path.join(output_dir, f"lut_layer_{li}.vhd")
        print(f"  Writing {layer_path} ... ({layer.output_size} neurons, n={layer.n})")
        with open(layer_path, "w") as f:
            emit_lut_layer(f, li, lut_contents, input_indices, layer_input_width)

        file_size = os.path.getsize(layer_path)
        print(f"    {file_size:,} bytes")
        layer_input_width = layer.output_size  # next layer input width

    # --- 3. GroupSum ---
    last_layer_size = hidden_sizes[-1]
    neurons_per_class = last_layer_size // num_classes
    gs_path = os.path.join(output_dir, "groupsum.vhd")
    print(f"  Writing {gs_path} ... ({num_classes} groups x {neurons_per_class} bits)")
    with open(gs_path, "w") as f:
        emit_groupsum(f, num_classes, neurons_per_class)

    # --- 4. Top-level ---
    top_path = os.path.join(output_dir, "dwn_top.vhd")
    print(f"  Writing {top_path} {'(pipelined)' if args.pipelined else '(combinational)'} ...")
    with open(top_path, "w") as f:
        if args.pipelined:
            emit_dwn_top_pipelined(f, input_features, num_bits, input_bits, hidden_sizes, num_classes)
        else:
            emit_dwn_top(f, input_features, num_bits, input_bits, hidden_sizes, num_classes)

    # Summary
    print(f"\nFloPoCo-style VHDL emitted to: {os.path.abspath(output_dir)}")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        sz = os.path.getsize(fpath)
        print(f"  {fname:30s}  {sz:>10,} bytes")

    print(f"\nSynthesis command:")
    print(f"  vivado -mode batch -source scripts/synth_dwn_vhdl.tcl \\")
    print(f"         -tclargs {os.path.abspath(output_dir)} "
          f"{os.path.abspath(output_dir)}/synth_results")


if __name__ == "__main__":
    main()

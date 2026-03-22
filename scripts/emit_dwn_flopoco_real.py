#!/usr/bin/env python3
"""
Generate real FloPoCo VHDL for a trained DWN checkpoint.

Uses the FloPoCo tool running in Docker (flopoco-dwn:latest) to generate:
  1. GenericLut entities for each neuron (batched into groups)
  2. IntMultiAdder bitheap-based popcount for GroupSum
  3. Behavioral VHDL for thermometer encoding (no FloPoCo equivalent)
  4. Structural top-level connecting everything (with optional pipeline registers)

Usage:
    conda run -n plena2 python scripts/emit_dwn_flopoco_real.py --ckpt-name baseline_n6
    conda run -n plena2 python scripts/emit_dwn_flopoco_real.py --ckpt /path/to/checkpoint.pt --no-pipelined

Output directory: mase_output/dwn/{ckpt_name}_flopoco_real/
"""
import argparse
import math
import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# sys.modules stubs to bypass heavy MASE imports
# ---------------------------------------------------------------------------
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, _src)

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
    p = argparse.ArgumentParser(
        description="Generate real FloPoCo VHDL for DWN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt-name", type=str, default="baseline_n6",
                    help="Checkpoint stem (looks in mase_output/dwn/<name>.pt)")
    p.add_argument("--ckpt", type=str, default=None,
                    help="Full checkpoint path (overrides --ckpt-name)")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: mase_output/dwn/<ckpt-name>_flopoco_real)")
    p.add_argument("--input-bits", type=int, default=8,
                    help="Bit width of raw input features (for threshold scaling)")
    p.add_argument("--pipelined", action="store_true", default=True,
                    help="Generate registered pipeline stages in dwn_top")
    p.add_argument("--no-pipelined", action="store_false", dest="pipelined",
                    help="Generate combinational dwn_top (no clk/rst)")
    p.add_argument("--batch-size", type=int, default=50,
                    help="Number of GenericLut operators per FloPoCo invocation")
    p.add_argument("--target", type=str, default="VirtexUltrascalePlus",
                    help="FloPoCo FPGA target")
    p.add_argument("--frequency", type=int, default=700,
                    help="FloPoCo target frequency in MHz")
    p.add_argument("--docker-image", type=str, default="flopoco-dwn:latest",
                    help="Docker image containing FloPoCo")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FloPoCo Docker helpers
# ---------------------------------------------------------------------------

def run_flopoco_docker(docker_image, flopoco_cmd, output_file_host, timeout=120):
    """
    Run a flopoco command inside Docker, writing VHDL to output_file_host.

    Returns stdout+stderr. Raises RuntimeError on failure.
    """
    host_dir = os.path.dirname(os.path.abspath(output_file_host))
    container_dir = "/output"
    container_file = os.path.join(container_dir, os.path.basename(output_file_host))

    cmd = [
        "docker", "run", "--rm",
        "--security-opt", "label=disable",
        "-v", f"{host_dir}:{container_dir}",
        "--entrypoint", "bash",
        docker_image,
        "-c", f"flopoco {flopoco_cmd} outputFile={container_file} 2>&1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(
            f"FloPoCo Docker failed (rc={result.returncode}):\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    if not os.path.exists(output_file_host):
        raise RuntimeError(
            f"FloPoCo did not produce output file: {output_file_host}\n"
            f"STDOUT: {result.stdout}"
        )

    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# GenericLut generation
# ---------------------------------------------------------------------------

def generate_lut_vhdl(docker_image, target, frequency, layer_idx, lut_contents,
                      batch_size, output_dir):
    """
    Generate FloPoCo GenericLut VHDL for all neurons in a layer.

    Neurons are batched into groups of batch_size to limit Docker invocations.
    Each batch produces one VHDL file containing multiple GenericLut entities.

    Returns:
        List of (base_entity_name, flopoco_entity_name) tuples for each neuron.
    """
    output_size = lut_contents.shape[0]
    num_entries = lut_contents.shape[1]
    n = int(math.log2(num_entries))

    # Precompute the inputValues string (same for all neurons with same n)
    input_values_str = ":".join(str(i) for i in range(num_entries))

    neuron_entities = []  # (base_entity_name, flopoco_entity_name)
    batch_files = []
    num_batches = (output_size + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, output_size)
        batch_count = end - start

        # Build flopoco command with multiple GenericLut operators
        parts = [f"target={target} frequency={frequency}"]

        for i, neuron_i in enumerate(range(start, end)):
            entity_base = f"lut_L{layer_idx}_N{neuron_i}"
            lut_row = lut_contents[neuron_i]
            output_values_str = ":".join(str(int(lut_row[j])) for j in range(num_entries))

            parts.append(
                f"GenericLut wIn={n} wOut=1 entityName={entity_base} "
                f"inputValues='{input_values_str}' outputValues='{output_values_str}'"
            )

            # FloPoCo entity name: GenericLut_{entityName}_Freq{freq}_uid{2+2*i}
            uid = 2 + 2 * i
            flopoco_name = f"GenericLut_{entity_base}_Freq{frequency}_uid{uid}"
            neuron_entities.append((entity_base, flopoco_name))

        flopoco_cmd = " ".join(parts)
        batch_file = os.path.join(output_dir, f"lut_layer_{layer_idx}_batch_{batch_idx}.vhd")
        batch_files.append(batch_file)

        print(f"    Batch {batch_idx + 1}/{num_batches}: neurons {start}-{end - 1} "
              f"({batch_count} neurons) ...", end=" ", flush=True)

        try:
            run_flopoco_docker(docker_image, flopoco_cmd, batch_file)
            sz = os.path.getsize(batch_file)
            print(f"OK ({sz:,} bytes)")
        except RuntimeError as e:
            print(f"FAILED")
            raise

    return neuron_entities, batch_files


# ---------------------------------------------------------------------------
# IntMultiAdder generation (popcount)
# ---------------------------------------------------------------------------

def generate_popcount_vhdl(docker_image, target, frequency, neurons_per_class,
                           output_dir):
    """
    Generate FloPoCo IntMultiAdder VHDL for the GroupSum popcount.

    Only one IntMultiAdder entity is needed (all class groups share it).

    Returns:
        (entity_name, vhdl_file, output_bits, has_clk) tuple.
    """
    flopoco_cmd = (
        f"target={target} frequency={frequency} "
        f"IntMultiAdder signedIn=0 n={neurons_per_class} wIn=1"
    )

    vhdl_file = os.path.join(output_dir, "popcount.vhd")
    print(f"  Generating IntMultiAdder (n={neurons_per_class}, wIn=1) ...", end=" ", flush=True)

    output = run_flopoco_docker(docker_image, flopoco_cmd, vhdl_file)

    # Parse entity name from FloPoCo output
    entity_name = f"IntMultiAdder_U1_{neurons_per_class}_Freq{frequency}_uid2"

    # Parse output width from the VHDL
    with open(vhdl_file, "r") as f:
        content = f.read()

    match = re.search(
        rf"entity\s+{re.escape(entity_name)}\s+is.*?"
        r"R\s*:\s*out\s+std_logic_vector\((\d+)\s+downto\s+0\)",
        content, re.DOTALL
    )
    if match:
        output_bits = int(match.group(1)) + 1
    else:
        output_bits = int(math.ceil(math.log2(neurons_per_class + 1)))
        print(f"WARNING: could not parse R width, assuming {output_bits}")

    has_clk = "clk : in std_logic" in content

    sz = os.path.getsize(vhdl_file)
    print(f"OK ({sz:,} bytes, entity={entity_name}, R={output_bits} bits, "
          f"{'pipelined' if has_clk else 'combinational'})")

    return entity_name, vhdl_file, output_bits, has_clk


# ---------------------------------------------------------------------------
# Thermometer encoder (behavioral VHDL, no FloPoCo equivalent)
# ---------------------------------------------------------------------------

def emit_thermometer_encoder(f, thresholds_tensor, num_bits, input_bits):
    """Generate thermometer_encoder.vhd (behavioral comparisons)."""
    num_features = thresholds_tensor.shape[0]
    total_out = num_features * num_bits
    max_val = (1 << input_bits) - 1

    f.write("-- Thermometer encoder: behavioral VHDL (no FloPoCo equivalent)\n")
    f.write("-- Generated by emit_dwn_flopoco_real.py\n\n")
    f.write("library ieee;\n")
    f.write("use ieee.std_logic_1164.all;\n")
    f.write("use ieee.numeric_std.all;\n\n")

    f.write(f"entity thermometer_encoder is\n")
    f.write(f"  port (\n")
    f.write(f"    x_in  : in  std_logic_vector({num_features * input_bits - 1} downto 0);\n")
    f.write(f"    x_out : out std_logic_vector({total_out - 1} downto 0)\n")
    f.write(f"  );\n")
    f.write(f"end entity thermometer_encoder;\n\n")

    f.write(f"architecture rtl of thermometer_encoder is\n")
    f.write(f"begin\n")
    f.write(f"  process(x_in)\n")
    f.write(f"    variable feat : unsigned({input_bits - 1} downto 0);\n")
    f.write(f"  begin\n")

    for feat_i in range(num_features):
        in_hi = (feat_i + 1) * input_bits - 1
        in_lo = feat_i * input_bits
        f.write(f"    feat := unsigned(x_in({in_hi} downto {in_lo}));\n")

        for bit_j in range(num_bits):
            raw_thresh = float(thresholds_tensor[feat_i, bit_j])
            thresh_int = int(round(raw_thresh * max_val))
            thresh_int = max(0, min(max_val, thresh_int))
            out_idx = feat_i * num_bits + bit_j

            f.write(f"    if feat >= to_unsigned({thresh_int + 1}, {input_bits}) then\n")
            f.write(f"      x_out({out_idx}) <= '1';\n")
            f.write(f"    else\n")
            f.write(f"      x_out({out_idx}) <= '0';\n")
            f.write(f"    end if;\n")

    f.write(f"  end process;\n")
    f.write(f"end architecture rtl;\n")


# ---------------------------------------------------------------------------
# Structural top-level
# ---------------------------------------------------------------------------

def _emit_register(f, src, dst, width, name):
    """Emit a clocked register process."""
    f.write(f"  -- Register: {name}\n")
    f.write(f"  process(clk)\n")
    f.write(f"  begin\n")
    f.write(f"    if rising_edge(clk) then\n")
    f.write(f"      if rst = '1' then\n")
    f.write(f"        {dst} <= (others => '0');\n")
    f.write(f"      else\n")
    f.write(f"        {dst} <= {src};\n")
    f.write(f"      end if;\n")
    f.write(f"    end if;\n")
    f.write(f"  end process;\n")


def emit_dwn_top(f, num_features, num_bits, input_bits, hidden_sizes, num_classes,
                 layer_neuron_entities, popcount_entity, popcount_bits,
                 popcount_has_clk, layer_input_indices, layer_lut_n,
                 pipelined, frequency):
    """
    Generate dwn_top.vhd connecting FloPoCo-generated components.

    - GenericLut ports are individual std_logic (i0..i5, o0), so we can
      directly map them to bit-selects of std_logic_vector signals.
    - IntMultiAdder ports X0..X99 are std_logic_vector(0 downto 0), so we
      need intermediate 1-bit signals for each input.
    """
    thermo_in_width = num_features * input_bits
    thermo_out_width = num_features * num_bits
    last_layer_size = hidden_sizes[-1]
    neurons_per_class = last_layer_size // num_classes
    total_out_bits = num_classes * popcount_bits
    num_layers = len(hidden_sizes)

    f.write("-- DWN top-level with real FloPoCo components\n")
    f.write(f"-- Generated by emit_dwn_flopoco_real.py\n")
    f.write(f"-- Target: {f'VirtexUltrascalePlus'} @ {frequency} MHz\n\n")
    f.write("library ieee;\n")
    f.write("use ieee.std_logic_1164.all;\n")
    f.write("use ieee.numeric_std.all;\n\n")

    f.write(f"entity dwn_top is\n")
    f.write(f"  port (\n")
    if pipelined or popcount_has_clk:
        f.write(f"    clk   : in  std_logic;\n")
        f.write(f"    rst   : in  std_logic;\n")
    f.write(f"    x_in  : in  std_logic_vector({thermo_in_width - 1} downto 0);\n")
    f.write(f"    y_out : out std_logic_vector({total_out_bits - 1} downto 0)\n")
    f.write(f"  );\n")
    f.write(f"end entity dwn_top;\n\n")

    f.write(f"architecture structural of dwn_top is\n\n")

    # -- Signal declarations --
    f.write(f"  -- Thermometer output\n")
    f.write(f"  signal thermo_comb : std_logic_vector({thermo_out_width - 1} downto 0);\n")
    if pipelined:
        f.write(f"  signal thermo_reg  : std_logic_vector({thermo_out_width - 1} downto 0);\n")
    f.write(f"\n")

    for li, sz in enumerate(hidden_sizes):
        f.write(f"  -- Layer {li} output ({sz} neurons)\n")
        f.write(f"  signal layer_{li}_comb : std_logic_vector({sz - 1} downto 0);\n")
        if pipelined:
            f.write(f"  signal layer_{li}_reg  : std_logic_vector({sz - 1} downto 0);\n")
    f.write(f"\n")

    # IntMultiAdder input signals: each X port is std_logic_vector(0 downto 0)
    # We need one signal per popcount input per class group
    f.write(f"  -- IntMultiAdder input signals (std_logic_vector(0 downto 0) each)\n")
    for g in range(num_classes):
        for i in range(neurons_per_class):
            f.write(f"  signal pc_{g}_x{i} : std_logic_vector(0 downto 0);\n")
    f.write(f"\n")

    # Popcount output signals
    f.write(f"  -- Popcount output signals (one per class group)\n")
    for g in range(num_classes):
        f.write(f"  signal popcount_{g}_R : std_logic_vector({popcount_bits - 1} downto 0);\n")
    f.write(f"\n")

    f.write(f"  -- GroupSum combined output\n")
    f.write(f"  signal gs_comb : std_logic_vector({total_out_bits - 1} downto 0);\n")
    if pipelined:
        f.write(f"  signal gs_reg  : std_logic_vector({total_out_bits - 1} downto 0);\n")
    f.write(f"\n")

    f.write(f"begin\n\n")

    # ---- Stage 0: Thermometer encoder ----
    f.write(f"  -- ===== Stage 0: Thermometer encoder =====\n")
    f.write(f"  thermo_inst: entity work.thermometer_encoder\n")
    f.write(f"    port map (x_in => x_in, x_out => thermo_comb);\n\n")

    if pipelined:
        _emit_register(f, "thermo_comb", "thermo_reg", thermo_out_width, "thermo")
        f.write(f"\n")

    # ---- LUT layers ----
    for li, sz in enumerate(hidden_sizes):
        n = layer_lut_n[li]
        input_indices = layer_input_indices[li]
        neuron_entities = layer_neuron_entities[li]

        if li == 0:
            in_sig = "thermo_reg" if pipelined else "thermo_comb"
        else:
            in_sig = f"layer_{li - 1}_reg" if pipelined else f"layer_{li - 1}_comb"

        f.write(f"  -- ===== Stage {li + 1}: LUT layer {li} ({sz} neurons, n={n}) =====\n")

        for neuron_i in range(sz):
            _, flopoco_name = neuron_entities[neuron_i]
            indices = input_indices[neuron_i]  # shape (n,)

            # GenericLut ports: i0..i(n-1) are std_logic, o0 is std_logic.
            # Port mapping to bit-select of std_logic_vector is valid VHDL.
            #
            # FloPoCo GenericLut internally does: t_in(0) <= i0; t_in(1) <= i1; ...
            # t_in is used as the LUT address.
            # Our input_indices are ordered MSB-first:
            #   indices[0] = MSB of LUT address = bit (n-1)
            #   indices[n-1] = LSB of LUT address = bit 0
            # So: i_k connects to t_in(k) = bit k of address = indices[n-1-k]
            port_maps = []
            for k in range(n):
                src_idx = int(indices[n - 1 - k])
                port_maps.append(f"i{k} => {in_sig}({src_idx})")
            port_maps.append(f"o0 => layer_{li}_comb({neuron_i})")

            f.write(f"  inst_L{li}_N{neuron_i}: entity work.{flopoco_name}\n")
            f.write(f"    port map ({', '.join(port_maps)});\n")

        f.write(f"\n")

        if pipelined:
            _emit_register(f, f"layer_{li}_comb", f"layer_{li}_reg", sz, f"layer_{li}")
            f.write(f"\n")

    # ---- GroupSum: IntMultiAdder popcount ----
    last_sig = f"layer_{num_layers - 1}_reg" if pipelined else f"layer_{num_layers - 1}_comb"

    f.write(f"  -- ===== Stage {num_layers + 1}: GroupSum popcount =====\n\n")

    # Wire IntMultiAdder input signals from last layer output
    f.write(f"  -- Wire popcount inputs from last layer output\n")
    for g in range(num_classes):
        for i in range(neurons_per_class):
            neuron_bit = g * neurons_per_class + i
            f.write(f"  pc_{g}_x{i}(0) <= {last_sig}({neuron_bit});\n")
    f.write(f"\n")

    # Instantiate popcount for each class group
    for g in range(num_classes):
        f.write(f"  -- Class {g}\n")
        f.write(f"  popcount_{g}: entity work.{popcount_entity}\n")
        f.write(f"    port map (\n")

        port_lines = []
        if popcount_has_clk:
            port_lines.append(f"      clk => clk")

        for i in range(neurons_per_class):
            port_lines.append(f"      X{i} => pc_{g}_x{i}")

        port_lines.append(f"      R => popcount_{g}_R")
        f.write(",\n".join(port_lines))
        f.write(f"\n    );\n")

        out_hi = (g + 1) * popcount_bits - 1
        out_lo = g * popcount_bits
        f.write(f"  gs_comb({out_hi} downto {out_lo}) <= "
                f"popcount_{g}_R({popcount_bits - 1} downto 0);\n\n")

    if pipelined:
        _emit_register(f, "gs_comb", "gs_reg", total_out_bits, "gs")
        f.write(f"\n")
        f.write(f"  y_out <= gs_reg;\n\n")
    else:
        f.write(f"  y_out <= gs_comb;\n\n")

    f.write(f"end architecture structural;\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    ckpt_dir = os.path.join(os.path.dirname(__file__), "../mase_output/dwn")
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")
    ckpt_name = args.ckpt_name if not args.ckpt else os.path.splitext(os.path.basename(args.ckpt))[0]
    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{ckpt_name}_flopoco_real")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Check Docker is available
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ERROR: Docker is not available. FloPoCo runs inside Docker.")
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

    # Collect layer metadata
    layer_lut_contents = []
    layer_input_indices = []
    layer_lut_n = []

    for li, layer in enumerate(lut_layers):
        lut_contents = layer.get_lut_contents().cpu()    # (output_size, 2^n)
        input_indices = layer.get_input_indices().cpu()   # (output_size, n)
        layer_lut_contents.append(lut_contents)
        layer_input_indices.append(input_indices)
        layer_lut_n.append(layer.n)
        print(f"  Layer {li}: {layer.output_size} neurons, n={layer.n}, "
              f"lut_entries={lut_contents.shape[1]}")

    # --- 1. Thermometer encoder (behavioral VHDL) ---
    thermo_path = os.path.join(output_dir, "thermometer_encoder.vhd")
    print(f"\nGenerating thermometer encoder ...")
    with open(thermo_path, "w") as f:
        emit_thermometer_encoder(f, thresholds, num_bits, input_bits)
    sz = os.path.getsize(thermo_path)
    print(f"  Written: {thermo_path} ({sz:,} bytes)")

    # --- 2. GenericLut entities for each layer ---
    all_layer_entities = []

    for li in range(num_layers):
        print(f"\nGenerating GenericLut VHDL for layer {li} "
              f"({hidden_sizes[li]} neurons, n={layer_lut_n[li]}) ...")

        neuron_entities, batch_files = generate_lut_vhdl(
            args.docker_image, args.target, args.frequency,
            li, layer_lut_contents[li], args.batch_size, output_dir,
        )
        all_layer_entities.append(neuron_entities)
        print(f"  Total: {len(neuron_entities)} neurons in {len(batch_files)} batch files")

    # --- 3. IntMultiAdder popcount ---
    last_layer_size = hidden_sizes[-1]
    neurons_per_class = last_layer_size // num_classes
    print(f"\nGenerating IntMultiAdder popcount ...")

    popcount_entity, popcount_file, popcount_bits, popcount_has_clk = \
        generate_popcount_vhdl(
            args.docker_image, args.target, args.frequency,
            neurons_per_class, output_dir,
        )

    # --- 4. Top-level ---
    top_path = os.path.join(output_dir, "dwn_top.vhd")
    mode_str = "pipelined" if args.pipelined else "combinational"
    print(f"\nGenerating top-level ({mode_str}) ...")

    with open(top_path, "w") as f:
        emit_dwn_top(
            f, input_features, num_bits, input_bits, hidden_sizes, num_classes,
            all_layer_entities, popcount_entity, popcount_bits,
            popcount_has_clk, layer_input_indices, layer_lut_n,
            args.pipelined, args.frequency,
        )
    sz = os.path.getsize(top_path)
    print(f"  Written: {top_path} ({sz:,} bytes)")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"FloPoCo VHDL emitted to: {os.path.abspath(output_dir)}")
    print(f"{'=' * 70}")

    total_bytes = 0
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".vhd"):
            continue
        fpath = os.path.join(output_dir, fname)
        fsz = os.path.getsize(fpath)
        total_bytes += fsz
        print(f"  {fname:50s}  {fsz:>10,} bytes")

    print(f"  {'TOTAL':50s}  {total_bytes:>10,} bytes")

    print(f"\nKey differences from hand-written VHDL:")
    print(f"  - GenericLut: FloPoCo's exact entity style (with...select, entity boundaries)")
    print(f"  - IntMultiAdder: bitheap compressor tree (NOT sequential accumulator)")
    print(f"  - IntMultiAdder is {'pipelined (1 cycle)' if popcount_has_clk else 'combinational'}")
    print(f"  - Total pipeline stages: {num_layers + 2 if args.pipelined else 0}"
          f"{' + IntMultiAdder internal' if popcount_has_clk else ''}")


if __name__ == "__main__":
    main()

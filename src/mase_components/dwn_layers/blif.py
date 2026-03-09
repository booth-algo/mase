"""
BLIF export for DWN LUT layers.

Enables post-training Boolean minimisation via ABC.

Usage:
    from mase_components.dwn_layers.blif import emit_layer_blif, emit_network_blif
    emit_network_blif(model, "network.blif")
"""
from pathlib import Path

import torch


def emit_layer_blif(layer, layer_idx: int, layer_input_offset: int, f) -> None:
    """
    Write .names blocks for all neurons in one LUT layer.

    Args:
        layer:              LUTLayer instance (eval mode, on any device).
        layer_idx:          Index of this layer (used to name output wires l{L}_out_{i}).
        layer_input_offset: Not used for wire naming — wire names are resolved by
                            get_input_indices(), which already holds global indices
                            relative to the layer's own input vector.  The caller
                            passes the appropriate input wire name list.
        f:                  Open file object to write into.
    """
    raise NotImplementedError("Use emit_layer_blif_with_names instead")


def _input_wire_names_for_layer(layer_idx: int, layer, prev_wire_names: list) -> list:
    """
    Return the list of wire name strings that index into the input of this layer.

    For layer 0 the inputs come from the thermometer encoder: x_0 … x_{N-1}.
    For layer L > 0 the inputs are the outputs of layer L-1: l{L-1}_out_0 … l{L-1}_out_{M-1}.

    prev_wire_names is the full list of wires available at this layer's input.
    """
    return prev_wire_names


def emit_network_blif(model, output_path) -> None:
    """
    Write the full DWN network as a single BLIF file.

    Only the LUT layers are emitted as Boolean functions (.names blocks).
    GroupSum is a sum operation and is excluded.

    Wire naming:
      - Thermometer-encoded inputs:  x_0 … x_{total_inputs-1}
      - Layer L output neuron i:     l{L}_out_{i}
      - Final layer outputs aliased: y_0 … y_{num_outputs-1}

    Args:
        model:       DWNModel (or any nn.Module with a .lut_layers ModuleList).
        output_path: Path-like; file will be created/overwritten.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lut_layers = list(model.lut_layers)
    if not lut_layers:
        raise ValueError("model.lut_layers is empty")

    # Total thermometer-encoded input width (layer 0's input_size)
    total_inputs = lut_layers[0].input_size

    # Primary inputs
    input_wire_names = [f"x_{i}" for i in range(total_inputs)]

    # Final outputs (last layer neurons)
    last_layer = lut_layers[-1]
    num_outputs = last_layer.output_size
    output_wire_names = [f"y_{i}" for i in range(num_outputs)]

    with output_path.open("w") as f:
        f.write(".model dwn\n")
        f.write(".inputs " + " ".join(input_wire_names) + "\n")
        f.write(".outputs " + " ".join(output_wire_names) + "\n")
        f.write("\n")

        # Track the wire names available at each layer's input
        current_input_wires = input_wire_names  # list indexed [0..input_size-1]

        for layer_idx, layer in enumerate(lut_layers):
            lut_contents = layer.get_lut_contents()   # (output_size, 2^n) int tensor
            input_indices = layer.get_input_indices()  # (output_size, n)  int32 tensor

            # Move to CPU for iteration
            lut_contents = lut_contents.cpu()
            input_indices = input_indices.cpu()

            output_size = layer.output_size
            n = layer.n
            is_last = (layer_idx == len(lut_layers) - 1)

            for neuron_i in range(output_size):
                # Wire names for this neuron's inputs
                in_wires = [
                    current_input_wires[int(input_indices[neuron_i, k])]
                    for k in range(n)
                ]

                # Output wire name
                out_wire = f"l{layer_idx}_out_{neuron_i}"

                f.write(f".names {' '.join(in_wires)} {out_wire}\n")

                # On-set entries: rows where lut_contents[neuron_i, j] == 1
                lut_row = lut_contents[neuron_i]  # shape (2^n,)
                for j in range(2 ** n):
                    if int(lut_row[j]) == 1:
                        # Binary address: j encoded in n bits, MSB = input 0
                        bits = format(j, f"0{n}b")
                        f.write(f"{bits} 1\n")

                f.write("\n")

            # Build output wire list for next layer
            current_input_wires = [f"l{layer_idx}_out_{i}" for i in range(output_size)]

        # Alias final layer outputs as y_0 … y_{num_outputs-1}
        for i in range(num_outputs):
            src = f"l{len(lut_layers)-1}_out_{i}"
            dst = f"y_{i}"
            # BLIF buffer: .names src dst\n1 1
            f.write(f".names {src} {dst}\n")
            f.write("1 1\n")
            f.write("\n")

        f.write(".end\n")

    print(f"BLIF written to: {output_path}")

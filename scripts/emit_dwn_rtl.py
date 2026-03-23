#!/usr/bin/env python3
"""
Emit synthesisable SystemVerilog from a trained DWN checkpoint.

Usage:
    python scripts/emit_dwn_rtl.py --ckpt-name baseline_n6 --output-dir hw/baseline_n6
    python scripts/emit_dwn_rtl.py --ckpt-name mixed_n6_2  --output-dir hw/mixed_n6_2

Output directory will contain:
    rtl/dwn_top.sv
    rtl/fixed_dwn_lut_layer.sv
    rtl/fixed_dwn_lut_neuron.sv
    (and other RTL dependencies)
"""
import argparse
import os
import re
import sys
import types

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, _src)

#
# Import-chain workaround
#
# The dependency chain  chop → chop.nn.quantized → transformers → torchvision
# crashes at runtime because of a torchvision/torch version mismatch
# (torchvision::nms kernel not registered).  RTL emission needs neither
# torchvision nor HuggingFace transformers, so we intercept ALL imports of
# these packages via a sys.meta_path finder and return harmless stub modules.
#

class _StubModule(types.ModuleType):
    """Stub module: silently absorbs attribute access and sub-module imports."""
    _ALLOW_DUNDER = frozenset({
        "__name__", "__package__", "__path__", "__spec__", "__loader__",
        "__file__", "__cached__", "__doc__", "__builtins__", "__version__",
    })

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Return a callable no-op object for class/function stubs
        class _Noop:
            def __init__(self, *a, **kw): pass
            def __call__(self, *a, **kw): return None
            def __getattr__(self, n): return _Noop()
            def __repr__(self): return f"<Stub {self.__class__.__name__}>"
        return _Noop


class _StubFinder:
    """
    sys.meta_path finder that intercepts ALL torchvision.* and transformers.*
    imports and replaces them with _StubModule instances.
    """
    _STUB_PREFIXES = ("torchvision", "transformers", "optimum")

    def find_module(self, fullname, path=None):
        for pfx in self._STUB_PREFIXES:
            if fullname == pfx or fullname.startswith(pfx + "."):
                return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        m = _StubModule(fullname)
        m.__path__ = []
        m.__package__ = fullname.rsplit(".", 1)[0] if "." in fullname else fullname
        m.__version__ = "0.0.0"
        # __spec__ must not be None for importlib.util.find_spec checks
        try:
            import importlib.util as _iu
            m.__spec__ = _iu.spec_from_loader(fullname, loader=None)
        except Exception:
            pass
        sys.modules[fullname] = m
        return m


# Install the finder BEFORE importing torch or anything in chop
sys.meta_path.insert(0, _StubFinder())

# Pre-populate top-level stubs so "from torchvision import ..." doesn't
# accidentally hit the real (broken) package if it exists on sys.path
for _top in ("torchvision", "transformers", "optimum"):
    if _top not in sys.modules:
        _StubFinder().load_module(_top)

# Provide realistic stubs for the specific symbols mase_graph.py uses:
#   from transformers import PreTrainedModel
#   from transformers.utils.fx import symbolic_trace, HFTracer
class _PreTrainedModel:
    pass

class _HFTracer:
    pass

def _hf_symbolic_trace(model, *a, **kw):
    raise NotImplementedError("HuggingFace symbolic_trace not available in stub mode")

sys.modules["transformers"].PreTrainedModel = _PreTrainedModel
_tf_utils_fx = _StubFinder().load_module("transformers.utils.fx")
_tf_utils_fx.HFTracer = _HFTracer
_tf_utils_fx.symbolic_trace = _hf_symbolic_trace

#

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt-name", type=str, default="best",
                        help="Checkpoint stem (looks in mase_output/dwn/<name>.pt)")
    parser.add_argument("--ckpt",      type=str, default=None,
                        help="Full checkpoint path (overrides --ckpt-name)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for emitted RTL (default: mase_output/dwn/<ckpt-name>_rtl)")
    parser.add_argument("--emit-blif", action="store_true",
                        help="Also emit a BLIF file for ABC Boolean minimisation")
    parser.add_argument("--pipelined", action="store_true",
                        help="Also emit a pipelined (clocked) variant: dwn_top_clocked.sv")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_dir = os.path.join(os.path.dirname(__file__), "../mase_output/dwn")
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")
    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{args.ckpt_name}_rtl")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["model_config"]
    print(f"  Config: hidden_sizes={cfg['hidden_sizes']}, lut_n={cfg['lut_n']}, "
          f"num_bits={cfg['num_bits']}, mapping_first={cfg['mapping_first']}")

    from chop.nn.dwn import DWNModel, LUTLayer
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reconstruct model to get trained LUT layer weights
    # Strip training-only keys not accepted by DWNModel.__init__
    model_kwargs = {k: v for k, v in cfg.items()
                    if k not in ('area_lambda', 'lambda_reg')}
    model = DWNModel(**model_kwargs).to(device)

    # fit_thermometer must be called before load_state_dict so that the
    # thermometer.thresholds buffer is registered (otherwise load_state_dict
    # rejects it as an unexpected key).  The placeholder data is immediately
    # overwritten by the checkpoint values in the load_state_dict call below.
    model.fit_thermometer(torch.zeros(2, cfg["input_features"]).to(device))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build a hardware-only wrapper: just the LUT layers.
    # The thermometer encoding is a software preprocessing step (not hardware),
    # and GroupSum is a simple integer adder not requiring MASE RTL emission.
    # The hardware input is thermometer-encoded binary: shape (B, input_features * num_bits).
    class DWNHardwareCore(nn.Module):
        """LUT-layer stack only — the synthesisable hardware core."""
        def __init__(self, lut_layers):
            super().__init__()
            self.lut_layers = nn.ModuleList(lut_layers)

        def forward(self, x):
            for layer in self.lut_layers:
                x = layer(x)
            return x

    hw_model = DWNHardwareCore(list(model.lut_layers)).to(device)
    hw_model.eval()

    # Dummy input for the hardware core: already thermometer-encoded binary.
    thermo_width = cfg["input_features"] * cfg["num_bits"]
    dummy_input = torch.zeros(1, thermo_width).to(device)

    # Build MaseGraph from the hardware core only
    graph = MaseGraph(
        hw_model,
        custom_ops={
            "modules": {
                LUTLayer: {
                    "args": {"x": "data_in"},
                    "module": "fixed_dwn_lut_layer",
                    "dependence_files": [
                        "dwn_layers/rtl/fixed_dwn_lut_neuron.sv",
                        "dwn_layers/rtl/fixed_dwn_lut_layer.sv",
                    ],
                }
            },
            "functions": {},
        },
    )

    graph, _ = init_metadata_analysis_pass(graph)
    graph, _ = add_common_metadata_analysis_pass(
        graph,
        pass_args={"dummy_in": {"x": dummy_input}, "add_value": False},
    )
    graph, _ = add_hardware_metadata_analysis_pass(graph)
    graph, _ = dwn_hardware_metadata_pass(graph)

    # Emit RTL
    graph, _ = emit_verilog_top_transform_pass(
        graph,
        pass_args={"project_dir": output_dir, "top_name": "dwn_top"},
    )
    graph, _ = emit_internal_rtl_transform_pass(
        graph,
        pass_args={"project_dir": output_dir},
    )

    rtl_dir = os.path.join(output_dir, "hardware", "rtl")
    sv_files = [f for f in os.listdir(rtl_dir) if f.endswith(".sv")]
    print(f"\nEmitted RTL to: {rtl_dir}")
    for f in sorted(sv_files):
        print(f"  {f}")
    print(f"\nSynthesis command (on Vivado server):")
    print(f"  vivado -mode batch -source scripts/synth_dwn.tcl "
          f"-tclargs {os.path.abspath(rtl_dir)} {os.path.abspath(output_dir)}/synth_results")

    if args.emit_blif:
        from mase_components.dwn_layers.blif import emit_network_blif
        blif_path = os.path.join(output_dir, "network.blif")
        emit_network_blif(model, blif_path)
        print(f"\nBLIF written to: {os.path.abspath(blif_path)}")

    if args.pipelined:
        # 1. Read the emitted dwn_top.sv
        top_sv_path = os.path.join(rtl_dir, "dwn_top.sv")
        with open(top_sv_path, "r") as f:
            top_sv = f.read()

        # 2. Substitute module name and layer module references
        clocked_sv = re.sub(r'\bmodule dwn_top\b', 'module dwn_top_clocked', top_sv)
        clocked_sv = re.sub(r'\bfixed_dwn_lut_layer\b', 'fixed_dwn_lut_layer_clocked', clocked_sv)

        # 3. Write dwn_top_clocked.sv
        clocked_top_path = os.path.join(rtl_dir, "dwn_top_clocked.sv")
        with open(clocked_top_path, "w") as f:
            f.write(clocked_sv)

        # 4. Copy fixed_dwn_lut_layer_clocked.sv into the output RTL dir
        import shutil
        clocked_layer_src = os.path.join(
            os.path.dirname(__file__),
            "../src/mase_components/dwn_layers/rtl/fixed_dwn_lut_layer_clocked.sv",
        )
        clocked_layer_dst = os.path.join(rtl_dir, "fixed_dwn_lut_layer_clocked.sv")
        shutil.copy(clocked_layer_src, clocked_layer_dst)

        print(f"\nPipelined RTL emitted:")
        print(f"  {clocked_top_path}")
        print(f"  {clocked_layer_dst}")
        print(f"\nSynthesis command for clocked variant (on Vivado server):")
        print(f"  vivado -mode batch -source scripts/synth_dwn.tcl "
              f"-tclargs {os.path.abspath(rtl_dir)} "
              f"{os.path.abspath(output_dir)}/synth_results_clocked "
              f"xcvc1902-viva1596-3HP-e-S dwn_top_clocked")


if __name__ == "__main__":
    main()

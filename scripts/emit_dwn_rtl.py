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
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Also emit full pipeline wrappers (thermometer + LUT + groupsum)")
    parser.add_argument("--feature-width", type=int, default=8,
                        help="Bits per input feature for full-pipeline thermometer encoding")
    return parser.parse_args()


def main():
    args = parse_args()

    if hasattr(args, 'ckpt_name') and args.ckpt_name and not re.match(r'^[\w\-]+$', args.ckpt_name):
        print(f"ERROR: --ckpt-name must be alphanumeric/hyphens/underscores, got: {args.ckpt_name}")
        sys.exit(1)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "../mase_output/dwn")
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")
    output_dir = args.output_dir or os.path.join(ckpt_dir, f"{args.ckpt_name}_rtl")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")

    from dwn.emit import emit_dwn_rtl
    result = emit_dwn_rtl(
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        emit_pipelined=args.pipelined,
        emit_blif=args.emit_blif,
        full_pipeline=args.full_pipeline,
        feature_width=args.feature_width,
    )

    print(f"\nEmitted RTL to: {result['rtl_dir']}")
    for f in result['sv_files']:
        print(f"  {f}")
    print(f"\nSynthesis command (on Vivado server):")
    print(f"  vivado -mode batch -source scripts/synth_dwn.tcl "
          f"-tclargs {result['rtl_dir']} {result['output_dir']}/synth_results")


if __name__ == "__main__":
    main()

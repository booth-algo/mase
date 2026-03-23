"""
Pytest runner for the pipeline-depth probe.
Drives ONE MNIST transaction into dwn_paper_scope_sim_wrapper,
reads output at delays 1-8 cycles to locate the correct pipeline depth.
"""
import json, os, sys, types

_SRC  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src"))
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
for p in [_SRC, _REPO]:
    if p not in sys.path:
        sys.path.insert(0, p)
for _pkg in ["chop", "chop.nn"]:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_SRC, *_pkg.split("."))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

import torch
import cocotb_test.simulator as simulator

RTL_COMPONENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rtl"))
_DEFAULT_CKPT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mase_output/dwn/mnist_n2.pt"))
_DEFAULT_RTL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mase_output/dwn/mnist_n2_rtl/hardware/rtl"))
_DEFAULT_CACHE   = os.path.expanduser("~/.cache/dwn/mnist/mnist_features.pt")


def test_pipeline_depth_probe():
    from chop.nn.dwn.model import DWNModel

    ckpt_path = os.environ.get("DWN_CKPT",    _DEFAULT_CKPT)
    rtl_dir   = os.environ.get("DWN_RTL_DIR", _DEFAULT_RTL_DIR)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg  = ckpt["model_config"]
    model_kwargs = {k: v for k, v in cfg.items() if k not in ("area_lambda", "lambda_reg")}
    model = DWNModel(**model_kwargs)
    model.fit_thermometer(torch.zeros(2, cfg["input_features"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    lut_layers  = list(model.lut_layers)
    num_classes = cfg["num_classes"]

    def sw_forward(x_bits, layers):
        for layer in layers:
            indices  = layer.get_input_indices().tolist()
            contents = layer.get_lut_contents().tolist()
            out = []
            for i in range(layer.output_size):
                addr = sum(x_bits[indices[i][k]] << k for k in range(layer.n))
                out.append(int(contents[i][addr]))
            x_bits = out
        return x_bits

    def group_sum(lut_bits, nc):
        gs = len(lut_bits) // nc
        return [sum(lut_bits[g*gs:(g+1)*gs]) for g in range(nc)]

    # Take 10 MNIST samples (txn#0 is the one we verify)
    cached = torch.load(_DEFAULT_CACHE, map_location="cpu", weights_only=True)
    transactions = []
    for i in range(10):
        img_flat = cached["X"][60000 + i].unsqueeze(0).float()
        label    = int(cached["y"][60000 + i].item())
        with torch.no_grad():
            thermo      = model.thermometer(img_flat)
            thermo_bits = thermo[0].int().tolist()
        thermo_packed   = sum(b << bi for bi, b in enumerate(thermo_bits))
        lut_bits        = sw_forward(thermo_bits, lut_layers)
        expected_scores = group_sum(lut_bits, num_classes)
        transactions.append({
            "thermo_packed":   thermo_packed,
            "expected_scores": expected_scores,
            "label":           label,
            "sw_pred":         max(range(num_classes), key=lambda c: expected_scores[c]),
        })

    print(f"\n[probe] txn#0 expected scores: {transactions[0]['expected_scores']}")

    config_path = os.path.join(os.path.dirname(__file__), "dwn_probe_uvm_config.json")
    config = {
        "num_inputs":   cfg["input_features"] * cfg["num_bits"],
        "num_classes":  num_classes,
        "transactions": transactions,
    }
    with open(config_path, "w") as f:
        json.dump(config, f)
    os.environ["DWN_PAPER_SCOPE_UVM_CONFIG"] = config_path

    simulator.run(
        verilog_sources=[
            os.path.join(RTL_COMPONENT_DIR, "fixed_dwn_lut_neuron.sv"),
            os.path.join(RTL_COMPONENT_DIR, "fixed_dwn_lut_layer_clocked.sv"),
            os.path.join(RTL_COMPONENT_DIR, "fixed_dwn_groupsum_pipelined.sv"),
            os.path.join(RTL_COMPONENT_DIR, "dwn_paper_scope_sim_wrapper.sv"),
            os.path.join(rtl_dir, "dwn_top_clocked.sv"),
            os.path.join(rtl_dir, "dwn_top_paper_scope.sv"),
        ],
        toplevel="dwn_paper_scope_sim_wrapper",
        module="dwn_pipeline_depth_probe",
        simulator="verilator",
        waves=False,
        build_dir=os.path.join(os.path.dirname(__file__), "sim_build_probe"),
        python_search_path=[os.path.dirname(__file__)],
        extra_args=["--Wno-TIMESCALEMOD"],
        make_args=["CFG_CXXFLAGS_PCH_I=-include"],
    )

import cocotb
from cocotb.runner import get_runner
from chop.nn.dwn.model import DWNModel
from pathlib import Path
import os
import shutil
import argparse
from mase_components.dwn_layers.emit import emit_dwn_rtl

import torch

rtl_dir = Path(__file__).parent / 'gen_rtl'
rtl_path = rtl_dir / 'hardware/rtl'
model_dir = Path('mase_output/dwn/').resolve()

def top_tb_runner(model_name, full, no_emit):
    model_path = model_dir / f"{model_name}.pt"
    sim = os.getenv('SIM', 'verilator')
    runner = get_runner(sim) 

    model = torch.load(model_path, map_location='cpu', weights_only=True)
    cfg = model['model_config']
    model_kwargs = {k: v for k, v in cfg.items() if k not in ("area_lambda", "lambda_reg")}
    model = DWNModel(**model_kwargs)
    if not no_emit:
        if os.path.exists(rtl_dir):
            shutil.rmtree(rtl_dir)
        os.mkdir(rtl_dir)
        emit_dwn_rtl(ckpt_path=model_path, output_dir=rtl_dir, full_pipeline=True)
    runner.build(
        hdl_toplevel='full_pipeline_top' if full else 'dwn_top',
        # TODO: Switch back to this one RTL generation is finalized
        # verilog_sources=[
        #     rtl_path /'dwn_top.sv',
        #     # rtl_path /'dwn_top_clocked.sv',
        #     # rtl_path /'dwn_top_paper_scope.sv',
        #     rtl_path /'fixed_dwn_groupsum.sv',
        #     # rtl_path /'fixed_dwn_groupsum_pipelined.sv',
        #     rtl_path /'fixed_dwn_lut_layer.sv',
        #     # rtl_path /'fixed_dwn_lut_layer_clocked.sv',
        #     rtl_path /'fixed_dwn_lut_neuron.sv',
        #     rtl_path /'fixed_dwn_thermometer.sv',
        #     rtl_path /'full_pipeline_top.sv',
        #     # rtl_path /'full_pipeline_top_clocked.sv',
        # ],
        verilog_sources=list((rtl_path).glob('*.sv')),
        waves=True
    )
    
    runner.test(
        test_module='top_tb',
        hdl_toplevel='full_pipeline_top' if full else 'dwn_top',
        waves=True,
        extra_env={'MODEL_PATH': str(model_path), 'MODEL_MODE': 'full' if full else 'core'}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run the testbench with all RTL components (thermometer and groupsum)')
    parser.add_argument('--no-emit', action='store_true', help='Do not emit RTL')
    parser.add_argument('--model', help='Model checkpoint name')
    args = parser.parse_args()
    top_tb_runner(args.model, args.full, args.no_emit)
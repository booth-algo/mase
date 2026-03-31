from cocotb.runner import get_runner
from pathlib import Path
import os
import sysconfig
import shutil
import argparse
from mase_components.dwn_layers.emit import emit_dwn_rtl

# TODO: maybe move the generated RTL to mase_output ?
rtl_dir = Path(__file__).parent / 'gen_rtl'
rtl_path = rtl_dir / 'hardware/rtl'
model_dir = Path('mase_output/dwn/').resolve()

# Forward parent environment to allow cuda extension compilation
def forward_env():
    python_include_path = sysconfig.get_path('include') or ''
    env = {
        'CPATH': os.pathsep.join(part for part in (python_include_path, os.environ.get('CPATH')) if part),
        'C_INCLUDE_PATH': os.pathsep.join(part for part in (python_include_path, os.environ.get('C_INCLUDE_PATH')) if part),
        'CPLUS_INCLUDE_PATH': os.pathsep.join(part for part in (python_include_path, os.environ.get('CPLUS_INCLUDE_PATH')) if part),
    }
    env.update({key: value for key in ('CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'TORCH_EXTENSIONS_DIR') if (value := os.environ.get(key))})
    return env

def top_tb_runner(model_name, full, pipelined, no_emit):
    model_path = model_dir / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    sim = os.getenv('SIM', 'verilator')
    runner = get_runner(sim) 

    if not no_emit:
        if os.path.exists(rtl_dir):
            shutil.rmtree(rtl_dir)
        os.mkdir(rtl_dir)
        emit_dwn_rtl(ckpt_path=model_path, output_dir=rtl_dir, full_pipeline=full, emit_pipelined=pipelined)
    if full:
        hdl_toplevel = 'full_pipeline_top_clocked' if pipelined else 'full_pipeline_top'
    else:
        hdl_toplevel = 'dwn_top_clocked' if pipelined else 'dwn_top'
    runner.build(
        hdl_toplevel = hdl_toplevel,
        verilog_sources=list((rtl_path).glob('*.sv')),
        waves=True
    )
    
    runner.test(
        test_module='top_tb',
        hdl_toplevel = hdl_toplevel,
        waves=True,
        extra_env={'MODEL_PATH': str(model_path), 'MODEL_MODE': 'full' if full else 'core'} | forward_env()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run the testbench with all RTL components (thermometer and groupsum)')
    parser.add_argument('--pipelined', action='store_true', help='Run the testbench with the pipelined RTL components')
    parser.add_argument('--no-emit', action='store_true', help='Do not emit RTL')
    parser.add_argument('--model', required=True, help='Model checkpoint name')
    args = parser.parse_args()
    top_tb_runner(args.model, args.full, args.pipelined, args.no_emit)
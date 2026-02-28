"""Cocotb testbench for DWN LUT layer functional equivalence against a trained model.

Reads configuration from the DWN_EQUIV_CONFIG environment variable, which must
point to a JSON file containing:
  - input_size            (int)
  - output_size           (int)
  - lut_n                 (int)
  - input_indices_packed  (int)
  - lut_contents_packed   (int)

For input_size <= 12 all 2^input_size patterns are tested exhaustively.
For larger input sizes, 512 random patterns (seeded) are tested.
"""

import json
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.handle import HierarchyObject
from cocotb.triggers import ClockCycles, RisingEdge

from fixed_dwn_lut_layer_tb import LUTLayerConfig, LUTLayerSWModel, LUTLayerTx, Scoreboard


# ---------------------------------------------------------------------------
# Load config from environment variable
# ---------------------------------------------------------------------------

def _load_config() -> LUTLayerConfig:
    config_path = os.environ.get("DWN_EQUIV_CONFIG", "")
    if not config_path:
        raise RuntimeError("DWN_EQUIV_CONFIG environment variable is not set")
    with open(config_path, "r") as f:
        data = json.load(f)
    return LUTLayerConfig(
        lut_n=int(data["lut_n"]),
        num_inputs=int(data["input_size"]),
        num_outputs=int(data["output_size"]),
        input_indices_packed=int(data["input_indices_packed"]),
        lut_contents_packed=int(data["lut_contents_packed"]),
    )


# ---------------------------------------------------------------------------
# Infrastructure (mirrors fixed_dwn_lut_layer_tb.py)
# ---------------------------------------------------------------------------

async def clock_reset(dut: HierarchyObject) -> None:
    await cocotb.start(Clock(dut.clk, 3.2, units="ns").start())
    dut.rst.value = 1
    dut.data_in_0_valid.value = 0
    dut.data_out_0_ready.value = 1
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def driver(dut: HierarchyObject, tx: LUTLayerTx) -> None:
    """Assert data_in_0 for one clock cycle."""
    dut.data_in_0.value = tx.to_rtl_input()
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 1)   # output stable (combinational)
    dut.data_in_0_valid.value = 0


async def monitor(dut: HierarchyObject) -> int:
    """Capture packed data_out_0."""
    return int(dut.data_out_0.value)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_lut_layer_equiv(dut: HierarchyObject) -> None:
    """Functional equivalence: RTL output must exactly match SW LUT reference model."""
    cfg = _load_config()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    await clock_reset(dut)

    if cfg.num_inputs <= 12:
        # Exhaustive: all 2^input_size patterns
        patterns = range(1 << cfg.num_inputs)
        cocotb.log.info(
            f"[EQUIV] Exhaustive mode: {1 << cfg.num_inputs} patterns "
            f"(input_size={cfg.num_inputs})"
        )
    else:
        # Random: 512 patterns with fixed seed for reproducibility
        rng = random.Random(42)
        max_val = (1 << cfg.num_inputs) - 1
        patterns = [rng.randint(0, max_val) for _ in range(512)]
        cocotb.log.info(
            f"[EQUIV] Random mode: 512 patterns (input_size={cfg.num_inputs}, seed=42)"
        )

    for val in patterns:
        bits = [(val >> i) & 1 for i in range(cfg.num_inputs)]
        tx = LUTLayerTx(bits=bits)
        await driver(dut, tx)
        rtl_out = await monitor(dut)
        sw_out = sw.predict_packed(tx)
        assert rtl_out == sw_out, (
            f"[EQUIV] MISMATCH bits={bits}: "
            f"RTL={bin(rtl_out)} SW={bin(sw_out)} "
            f"(per_lut: {sw.predict(tx)})"
        )
        sb.passed += 1

    cocotb.log.info(
        f"[EQUIV] PASS: {sb.passed} patterns matched SW reference model "
        f"(input_size={cfg.num_inputs}, output_size={cfg.num_outputs}, lut_n={cfg.lut_n})"
    )

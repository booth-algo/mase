"""Cocotb testbench for dwn_top full-network functional equivalence.

Reads configuration from DWN_TOP_EQUIV_CONFIG env var pointing to a JSON file:
  - num_inputs:    int  (total input bits = INPUT_SIZE of first layer)
  - num_outputs:   int  (total output bits = OUTPUT_SIZE of last layer)
  - test_vectors:  list of {"input_packed": int, "output_packed": int}
"""

import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.handle import HierarchyObject
from cocotb.triggers import ClockCycles, RisingEdge


def _load_config():
    config_path = os.environ.get("DWN_TOP_EQUIV_CONFIG", "")
    if not config_path:
        raise RuntimeError("DWN_TOP_EQUIV_CONFIG environment variable is not set")
    with open(config_path) as f:
        return json.load(f)


async def clock_reset(dut):
    await cocotb.start(Clock(dut.clk, 3.2, units="ns").start())
    dut.rst.value = 1
    dut.data_in_0_valid.value = 0
    dut.data_out_0_ready.value = 1
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_dwn_top_equiv(dut):
    """Drive test vectors through dwn_top RTL and compare against SW golden model."""
    cfg = _load_config()
    test_vectors = cfg["test_vectors"]

    await clock_reset(dut)

    passed = 0
    for vec in test_vectors:
        inp = vec["input_packed"]
        expected = vec["output_packed"]

        dut.data_in_0.value = inp
        dut.data_in_0_valid.value = 1
        dut.data_out_0_ready.value = 1
        await RisingEdge(dut.clk)
        await ClockCycles(dut.clk, 1)   # combinational settle
        dut.data_in_0_valid.value = 0

        actual = int(dut.data_out_0.value)
        assert actual == expected, (
            f"MISMATCH input={inp:#x}: RTL={actual:#x} SW={expected:#x}"
        )
        passed += 1

    cocotb.log.info(
        f"[dwn_top_equiv] PASS: {passed}/{len(test_vectors)} vectors matched "
        f"(num_inputs={cfg['num_inputs']}, num_outputs={cfg['num_outputs']})"
    )

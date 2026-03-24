"""
Minimal cocotb probe: drives ONE transaction, reads output at delays 1..8,
prints which delay gives the correct group-sum scores.

Helps find the actual pipeline depth of dwn_top_paper_scope.

Usage:
    DWN_PAPER_SCOPE_UVM_CONFIG=<config.json> cocotb-run ...
    (set automatically by test_dwn_pipeline_depth_probe.py)
"""
import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge


@cocotb.test()
async def test_pipeline_depth(dut):
    """
    Streaming pipeline depth test.

    Drives txn#0 on E1, txn#1 on E2, ..., txn#K on E_K+1 (one new txn per cycle).
    After E1 (when txn#0 enters FF1), waits N extra edges and reads output.
    We expect txn#0's result to appear after exactly PIPELINE_DEPTH-1 extra cycles.
    """
    config_path = os.environ.get("DWN_PAPER_SCOPE_UVM_CONFIG", "")
    with open(config_path) as f:
        config = json.load(f)

    txns             = config["transactions"]
    nc               = config.get("num_classes", 10)
    txn0_expected    = txns[0]["expected_scores"]

    await cocotb.start(Clock(dut.clk, 3.2, units="ns").start())
    dut.rst.value = 1
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    await RisingEdge(dut.clk)   # pre-start edge

    # Drive txn#0 → E1 (enters FF1)
    dut.data_in_0.value = txns[0]["thermo_packed"]
    await RisingEdge(dut.clk)   # E1: txn#0 captured in FF1

    # Continue streaming txn#1, 2, ... while checking output
    found_match = False
    for extra in range(1, 9):
        # Drive next transaction
        if extra < len(txns):
            dut.data_in_0.value = txns[extra]["thermo_packed"]
        await RisingEdge(dut.clk)   # E_{extra+1}

        flat_val   = int(dut.data_out_0_flat.value)
        rtl_scores = [(flat_val >> (i * 8)) & 0xFF for i in range(nc)]
        match = (rtl_scores == txn0_expected)
        if match:
            found_match = True
        cocotb.log.info(
            f"  extra_wait={extra}:  rtl={rtl_scores}  "
            f"exp={txn0_expected}  match={match}"
        )

    assert found_match, f"No pipeline delay 1..8 produced the expected output for transaction #0"

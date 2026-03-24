"""Cocotb testbench for structural_dwn_lut_layer.

The structural variant has no clk/rst ports (purely combinational),
so this testbench uses Timer-based delays instead of clock edges.
The test logic mirrors fixed_dwn_lut_layer_tb exactly.
"""

import cocotb
from cocotb.handle import HierarchyObject
from cocotb.triggers import Timer

from dwn_lut_layer_common import LUTLayerConfig, LUTLayerTx, LUTLayerSWModel, Scoreboard


# Infrastructure (no clock -- purely combinational)

async def drive_and_settle(dut: HierarchyObject, tx: LUTLayerTx) -> None:
    """Assert data_in_0 and wait for combinational propagation."""
    dut.data_in_0.value = tx.to_rtl_input()
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    await Timer(2, units="ns")


async def monitor(dut: HierarchyObject) -> int:
    """Capture packed data_out_0."""
    return int(dut.data_out_0.value)


# Tests

@cocotb.test()
async def test_lut_basic(dut: HierarchyObject) -> None:
    """4'b1010: LUT0 addr=2->1, LUT1 addr=2->0; output=0b01 (SW agrees)."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    tx = LUTLayerTx(bits=[0, 1, 0, 1])
    await drive_and_settle(dut, tx)
    sb.check(tx, await monitor(dut), label="basic")


@cocotb.test()
async def test_lut_all_zeros(dut: HierarchyObject) -> None:
    """4'b0000: both LUTs at addr=0 -> out=0b00."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    tx = LUTLayerTx(bits=[0, 0, 0, 0])
    await drive_and_settle(dut, tx)
    sb.check(tx, await monitor(dut), label="all_zeros")


@cocotb.test()
async def test_lut_all_ones(dut: HierarchyObject) -> None:
    """4'b1111: both LUTs at addr=3 -> out=0b11."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    tx = LUTLayerTx(bits=[1, 1, 1, 1])
    await drive_and_settle(dut, tx)
    sb.check(tx, await monitor(dut), label="all_ones")


@cocotb.test()
async def test_lut_valid_ready(dut: HierarchyObject) -> None:
    """Valid/ready pass-through: output valid tracks input valid combinationally."""
    dut.data_in_0.value = 0
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    await Timer(2, units="ns")
    assert int(dut.data_out_0_valid.value) == 1, "data_out_0_valid should be 1"
    assert int(dut.data_in_0_ready.value) == 1, "data_in_0_ready should be 1"


@cocotb.test()
async def test_lut_exhaustive(dut: HierarchyObject) -> None:
    """Exhaustive: all 16 4-bit input patterns vs PyTorch LUT reference model."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    for val in range(1 << cfg.num_inputs):
        bits = [(val >> i) & 1 for i in range(cfg.num_inputs)]
        tx = LUTLayerTx(bits=bits)
        await drive_and_settle(dut, tx)
        sb.check(tx, await monitor(dut), label=f"exhaust[{val:04b}]")

    cocotb.log.info(f"[SCOREBOARD] total: {sb.passed} passed, {sb.failed} failed")

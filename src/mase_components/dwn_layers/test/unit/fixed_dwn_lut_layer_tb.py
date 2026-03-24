"""Cocotb testbench for fixed_dwn_lut_layer and structural_dwn_lut_layer.

Works for both clocked (fixed) and combinational (structural) DUT variants.
The DUT type is detected at runtime by checking for the presence of a clk signal.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.handle import HierarchyObject
from cocotb.triggers import ClockCycles, RisingEdge, Timer

from dwn_lut_layer_common import LUTLayerConfig, LUTLayerTx, LUTLayerSWModel, Scoreboard


# Infrastructure

def _is_clocked(dut):
    """Return True if the DUT has a clock signal (clocked variant)."""
    return hasattr(dut, 'clk')


async def init_dut(dut: HierarchyObject) -> None:
    """Reset clocked DUTs or initialise combinational DUTs."""
    if _is_clocked(dut):
        await cocotb.start(Clock(dut.clk, 3.2, units="ns").start())
        dut.rst.value = 1
        dut.data_in_0_valid.value = 0
        dut.data_out_0_ready.value = 1
        await ClockCycles(dut.clk, 2)
        dut.rst.value = 0
        await RisingEdge(dut.clk)


async def driver(dut: HierarchyObject, tx: LUTLayerTx) -> None:
    """Assert data_in_0 and wait for output to settle."""
    dut.data_in_0.value = tx.to_rtl_input()
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    if _is_clocked(dut):
        await RisingEdge(dut.clk)
        await ClockCycles(dut.clk, 1)   # output stable (combinational)
    else:
        await Timer(2, units="ns")
    dut.data_in_0_valid.value = 0


async def _wait(dut: HierarchyObject) -> None:
    """Wait one step appropriate for the DUT type."""
    if _is_clocked(dut):
        await RisingEdge(dut.clk)
        await ClockCycles(dut.clk, 1)
    else:
        await Timer(2, units="ns")


async def monitor(dut: HierarchyObject) -> int:
    """Capture packed data_out_0."""
    return int(dut.data_out_0.value)


# Tests (sequences)

@cocotb.test()
async def test_lut_basic(dut: HierarchyObject) -> None:
    """4'b1010: LUT0 addr=2->1, LUT1 addr=2->0; output=0b01 (SW agrees)."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    await init_dut(dut)
    # bits=[0,1,0,1]: LUT0 reads [0,1] addr=2 table=0b1100 out=1
    #                 LUT1 reads [0,1] addr=2 table=0b1010 out=0
    tx = LUTLayerTx(bits=[0, 1, 0, 1])
    await driver(dut, tx)
    sb.check(tx, await monitor(dut), label="basic")


@cocotb.test()
async def test_lut_all_zeros(dut: HierarchyObject) -> None:
    """4'b0000: both LUTs at addr=0 -> out=0b00."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    await init_dut(dut)
    tx = LUTLayerTx(bits=[0, 0, 0, 0])
    await driver(dut, tx)
    sb.check(tx, await monitor(dut), label="all_zeros")


@cocotb.test()
async def test_lut_all_ones(dut: HierarchyObject) -> None:
    """4'b1111: both LUTs at addr=3 -> out=0b11."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    await init_dut(dut)
    tx = LUTLayerTx(bits=[1, 1, 1, 1])
    await driver(dut, tx)
    sb.check(tx, await monitor(dut), label="all_ones")


@cocotb.test()
async def test_lut_valid_ready(dut: HierarchyObject) -> None:
    """Valid/ready pass-through: output valid tracks input valid combinationally."""
    await init_dut(dut)
    dut.data_in_0.value = 0
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    await _wait(dut)
    assert int(dut.data_out_0_valid.value) == 1, "data_out_0_valid should be 1"
    assert int(dut.data_in_0_ready.value) == 1, "data_in_0_ready should be 1"


@cocotb.test()
async def test_lut_exhaustive(dut: HierarchyObject) -> None:
    """Exhaustive: all 16 4-bit input patterns vs PyTorch LUT reference model."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    await init_dut(dut)
    for val in range(1 << cfg.num_inputs):
        bits = [(val >> i) & 1 for i in range(cfg.num_inputs)]
        tx = LUTLayerTx(bits=bits)
        await driver(dut, tx)
        sb.check(tx, await monitor(dut), label=f"exhaust[{val:04b}]")

    cocotb.log.info(f"[SCOREBOARD] total: {sb.passed} passed, {sb.failed} failed")

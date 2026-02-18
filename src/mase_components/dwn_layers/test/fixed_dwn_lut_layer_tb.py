"""Cocotb testbench for fixed_dwn_lut_layer"""

import random
from dataclasses import dataclass
from typing import List

import cocotb
from cocotb.clock import Clock
from cocotb.handle import HierarchyObject
from cocotb.triggers import ClockCycles, RisingEdge
import torch


# Config

@dataclass
class LUTLayerConfig:
    lut_n: int = 2                       # LUT_N: inputs per LUT neuron
    num_inputs: int = 4                   # INPUT_SIZE
    num_outputs: int = 2                  # OUTPUT_SIZE
    input_indices_packed: int = 0x03020100  # INPUT_INDICES RTL parameter
    lut_contents_packed: int = 0xAC         # LUT_CONTENTS RTL parameter

    def input_indices(self) -> List[List[int]]:
        """Unpack INPUT_INDICES: indices[i][k] = which input bit LUT i reads at position k.
        RTL packing: INPUT_INDICES[(i*LUT_N + k)*8 +: 8]
        """
        result = []
        for i in range(self.num_outputs):
            row = []
            for k in range(self.lut_n):
                byte_pos = (i * self.lut_n + k) * 8
                row.append((self.input_indices_packed >> byte_pos) & 0xFF)
            result.append(row)
        return result

    def lut_tables(self) -> List[torch.Tensor]:
        """Unpack LUT_CONTENTS into per-neuron lookup tables (2^LUT_N entries each).
        RTL packing: LUT_CONTENTS[i*(2**LUT_N) +: (2**LUT_N)]
        """
        entries = 1 << self.lut_n
        tables = []
        for i in range(self.num_outputs):
            raw = (self.lut_contents_packed >> (i * entries)) & ((1 << entries) - 1)
            bits = [(raw >> addr) & 1 for addr in range(entries)]
            tables.append(torch.tensor(bits, dtype=torch.int32))
        return tables


# Transaction (sequence item)

@dataclass
class LUTLayerTx:
    bits: List[int]  # binary input (0/1), length == num_inputs

    def to_rtl_input(self) -> int:
        """Pack bits into RTL integer (bit 0 → LSB)."""
        val = 0
        for i, b in enumerate(self.bits):
            val |= (b & 1) << i
        return val

    def to_torch(self) -> torch.Tensor:
        """Return int32 tensor (num_inputs,)."""
        return torch.tensor(self.bits, dtype=torch.int32)


# SW Golden Model — PyTorch tensor LUT lookup

class LUTLayerSWModel:
    """
    Reference model using PyTorch tensor indexing for LUT lookup.

    For each output neuron i:
      - Build address: addr = sum(bits[input_indices[i][k]] * 2^k, k=0..LUT_N-1)
      - Output:        out[i] = lut_tables[i][addr]

    This directly mirrors fixed_dwn_lut_neuron.sv:
        assign data_out_0 = LUT_CONTENTS[data_in_0];
    where data_in_0 is the concatenated input bits used as a table address.
    """

    def __init__(self, cfg: LUTLayerConfig):
        self.cfg = cfg
        self.indices = cfg.input_indices()   # list[list[int]], shape (num_out, lut_n)
        self.tables = cfg.lut_tables()       # list[Tensor(2^lut_n)], one per neuron
        self._powers = torch.tensor(
            [1 << k for k in range(cfg.lut_n)], dtype=torch.int32
        )

    def predict(self, tx: LUTLayerTx) -> List[int]:
        """Return list of output bits (one per LUT neuron)."""
        x = tx.to_torch()
        outputs = []
        for i in range(self.cfg.num_outputs):
            # Gather the input bits this LUT reads, form address
            addr_bits = torch.tensor(
                [x[self.indices[i][k]].item() for k in range(self.cfg.lut_n)],
                dtype=torch.int32,
            )
            addr = int((addr_bits * self._powers).sum().item())
            outputs.append(int(self.tables[i][addr].item()))
        return outputs

    def predict_packed(self, tx: LUTLayerTx) -> int:
        """Return outputs as packed integer (bit i = output of LUT neuron i)."""
        return sum(b << i for i, b in enumerate(self.predict(tx)))


# Scoreboard

class Scoreboard:
    def __init__(self, sw: LUTLayerSWModel):
        self.sw = sw
        self.passed = 0
        self.failed = 0

    def check(self, tx: LUTLayerTx, rtl_out: int, label: str = "") -> None:
        sw_out = self.sw.predict_packed(tx)
        sw_per_lut = self.sw.predict(tx)
        tag = f"[SCOREBOARD]{' ' + label if label else ''}"
        if rtl_out == sw_out:
            cocotb.log.info(
                f"{tag} PASS  bits={tx.bits} -> out={bin(rtl_out)} "
                f"per_lut={sw_per_lut}"
            )
            self.passed += 1
        else:
            self.failed += 1
            assert False, (
                f"{tag} bits={tx.bits}: RTL={bin(rtl_out)}, SW={bin(sw_out)} "
                f"(per_lut: {sw_per_lut})"
            )


# Infrastructure

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


# Tests (sequences)

@cocotb.test()
async def test_lut_basic(dut: HierarchyObject) -> None:
    """4'b1010: LUT0 addr=2->1, LUT1 addr=2->0; output=0b01 (SW agrees)."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    await clock_reset(dut)
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

    await clock_reset(dut)
    tx = LUTLayerTx(bits=[0, 0, 0, 0])
    await driver(dut, tx)
    sb.check(tx, await monitor(dut), label="all_zeros")


@cocotb.test()
async def test_lut_all_ones(dut: HierarchyObject) -> None:
    """4'b1111: both LUTs at addr=3 -> out=0b11."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    await clock_reset(dut)
    tx = LUTLayerTx(bits=[1, 1, 1, 1])
    await driver(dut, tx)
    sb.check(tx, await monitor(dut), label="all_ones")


@cocotb.test()
async def test_lut_valid_ready(dut: HierarchyObject) -> None:
    """Valid/ready pass-through: output valid tracks input valid combinationally."""
    await clock_reset(dut)
    dut.data_in_0.value = 0
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 1)
    assert int(dut.data_out_0_valid.value) == 1, "data_out_0_valid should be 1"
    assert int(dut.data_in_0_ready.value) == 1, "data_in_0_ready should be 1"


@cocotb.test()
async def test_lut_exhaustive(dut: HierarchyObject) -> None:
    """Exhaustive: all 16 4-bit input patterns vs PyTorch LUT reference model."""
    cfg = LUTLayerConfig()
    sw = LUTLayerSWModel(cfg)
    sb = Scoreboard(sw)

    await clock_reset(dut)
    for val in range(1 << cfg.num_inputs):
        bits = [(val >> i) & 1 for i in range(cfg.num_inputs)]
        tx = LUTLayerTx(bits=bits)
        await driver(dut, tx)
        sb.check(tx, await monitor(dut), label=f"exhaust[{val:04b}]")

    cocotb.log.info(f"[SCOREBOARD] total: {sb.passed} passed, {sb.failed} failed")

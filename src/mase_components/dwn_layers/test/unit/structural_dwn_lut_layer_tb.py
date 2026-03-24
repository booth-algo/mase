"""Cocotb testbench for structural_dwn_lut_layer.

The structural variant has no clk/rst ports (purely combinational),
so this testbench uses Timer-based delays instead of clock edges.
The test logic mirrors fixed_dwn_lut_layer_tb exactly.
"""

import random
from dataclasses import dataclass
from typing import List

import cocotb
from cocotb.handle import HierarchyObject
from cocotb.triggers import Timer
import torch


# Config

@dataclass
class LUTLayerConfig:
    lut_n: int = 2                       # LUT_N: inputs per LUT neuron
    num_inputs: int = 4                   # INPUT_SIZE
    num_outputs: int = 2                  # OUTPUT_SIZE
    input_indices_packed: int = 0xE4       # INPUT_INDICES RTL parameter (2-bit packed)
    lut_contents_packed: int = 0xAC        # LUT_CONTENTS RTL parameter

    @property
    def index_bits(self) -> int:
        """Number of bits per index field = ceil(log2(INPUT_SIZE))."""
        n = max(self.num_inputs, 2)
        return (n - 1).bit_length()

    def input_indices(self) -> List[List[int]]:
        """Unpack INPUT_INDICES: indices[i][k] = which input bit LUT i reads at position k.
        RTL packing: INPUT_INDICES[(i*LUT_N + k)*INDEX_BITS +: INDEX_BITS]
        """
        ib = self.index_bits
        mask = (1 << ib) - 1
        result = []
        for i in range(self.num_outputs):
            row = []
            for k in range(self.lut_n):
                bit_pos = (i * self.lut_n + k) * ib
                row.append((self.input_indices_packed >> bit_pos) & mask)
            result.append(row)
        return result

    def lut_tables(self) -> List[torch.Tensor]:
        """Unpack LUT_CONTENTS into per-neuron lookup tables (2^LUT_N entries each)."""
        entries = 1 << self.lut_n
        tables = []
        for i in range(self.num_outputs):
            raw = (self.lut_contents_packed >> (i * entries)) & ((1 << entries) - 1)
            bits = [(raw >> addr) & 1 for addr in range(entries)]
            tables.append(torch.tensor(bits, dtype=torch.int32))
        return tables


# Transaction

@dataclass
class LUTLayerTx:
    bits: List[int]  # binary input (0/1), length == num_inputs

    def to_rtl_input(self) -> int:
        """Pack bits into RTL integer (bit 0 -> LSB)."""
        val = 0
        for i, b in enumerate(self.bits):
            val |= (b & 1) << i
        return val

    def to_torch(self) -> torch.Tensor:
        return torch.tensor(self.bits, dtype=torch.int32)


# SW Golden Model

class LUTLayerSWModel:
    def __init__(self, cfg: LUTLayerConfig):
        self.cfg = cfg
        self.indices = cfg.input_indices()
        self.tables = cfg.lut_tables()
        self._powers = torch.tensor(
            [1 << k for k in range(cfg.lut_n)], dtype=torch.int32
        )

    def predict(self, tx: LUTLayerTx) -> List[int]:
        x = tx.to_torch()
        outputs = []
        for i in range(self.cfg.num_outputs):
            addr_bits = torch.tensor(
                [x[self.indices[i][k]].item() for k in range(self.cfg.lut_n)],
                dtype=torch.int32,
            )
            addr = int((addr_bits * self._powers).sum().item())
            outputs.append(int(self.tables[i][addr].item()))
        return outputs

    def predict_packed(self, tx: LUTLayerTx) -> int:
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


# Infrastructure (no clock — purely combinational)

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

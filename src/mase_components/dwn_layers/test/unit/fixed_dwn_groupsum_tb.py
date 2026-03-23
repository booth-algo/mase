"""Cocotb testbench for fixed_dwn_groupsum"""

import random
from dataclasses import dataclass
from typing import List

import cocotb
from cocotb.handle import HierarchyObject
from cocotb.triggers import Timer
import torch


# Config

@dataclass
class GroupSumConfig:
    input_size: int = 4   # INPUT_SIZE
    num_groups: int = 2   # NUM_GROUPS

    @property
    def group_size(self) -> int:
        return self.input_size // self.num_groups


# Transaction (sequence item)

@dataclass
class GroupSumTx:
    bits: List[int]  # binary input (0/1), length == input_size

    def to_rtl_input(self) -> int:
        """Pack bits into RTL integer (bit 0 → LSB)."""
        val = 0
        for i, b in enumerate(self.bits):
            val |= (b & 1) << i
        return val

    def to_torch(self) -> torch.Tensor:
        """Return float32 tensor (1, input_size) for PyTorch model."""
        return torch.tensor(self.bits, dtype=torch.float32).unsqueeze(0)


# SW Golden Model — PyTorch chop.nn.dwn.GroupSum

class GroupSumSWModel:
    """
    PyTorch reference model equivalent to chop.nn.dwn.GroupSum(k, tau=1.0).

    Inlines the GroupSum.forward() logic using PyTorch tensor ops to avoid
    importing the full chop package inside the cocotb subprocess:
        x.view(B, k, group_size).sum(dim=-1) / tau   (tau=1 → integer counts)

    This is bit-for-bit identical to the module in src/chop/nn/dwn/group_sum.py
    and matches $countones per group in the RTL.
    """

    def __init__(self, cfg: GroupSumConfig):
        self.k = cfg.num_groups
        self.group_size = cfg.group_size

    def predict(self, tx: GroupSumTx) -> List[int]:
        """Return integer popcount per group (matches GroupSum.forward, tau=1)."""
        x = tx.to_torch()  # (1, input_size) float
        counts = x.view(1, self.k, self.group_size).sum(dim=-1)  # (1, k)
        return [int(round(v.item())) for v in counts[0]]


# Scoreboard

class Scoreboard:
    def __init__(self, sw: GroupSumSWModel, cfg: GroupSumConfig):
        self.sw = sw
        self.cfg = cfg
        self.passed = 0
        self.failed = 0

    def check(self, tx: GroupSumTx, rtl_out: List[int], label: str = "") -> None:
        sw_out = self.sw.predict(tx)
        tag = f"[SCOREBOARD]{' ' + label if label else ''}"
        ok = all(rtl_out[g] == sw_out[g] for g in range(self.cfg.num_groups))
        if ok:
            cocotb.log.info(
                f"{tag} PASS  bits={tx.bits} -> counts={rtl_out}"
            )
            self.passed += 1
        else:
            self.failed += 1
            mismatches = [
                f"group{g}: RTL={rtl_out[g]}, SW={sw_out[g]}"
                for g in range(self.cfg.num_groups)
                if rtl_out[g] != sw_out[g]
            ]
            assert False, f"{tag} mismatch {', '.join(mismatches)}  bits={tx.bits}"


# Infrastructure — driver, monitor

async def init_dut(dut: HierarchyObject) -> None:
    dut.data_in_0_valid.value = 0
    dut.data_out_0_ready.value = 1
    await Timer(1, units="ns")


async def driver(dut: HierarchyObject, tx: GroupSumTx) -> None:
    dut.data_in_0.value = tx.to_rtl_input()
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    await Timer(1, units="ns")
    dut.data_in_0_valid.value = 0


async def monitor(dut: HierarchyObject, cfg: GroupSumConfig) -> List[int]:
    """Capture data_out_0 array (one per group)."""
    return [int(dut.data_out_0[g].value) for g in range(cfg.num_groups)]


# Tests (sequences)

@cocotb.test()
async def test_groupsum_basic(dut: HierarchyObject) -> None:
    """4'b1011 -> group0=2, group1=1 (cross-checked with PyTorch GroupSum)."""
    cfg = GroupSumConfig()
    sw = GroupSumSWModel(cfg)
    sb = Scoreboard(sw, cfg)

    await init_dut(dut)
    tx = GroupSumTx(bits=[1, 1, 0, 1])  # group0=[1,1]=2, group1=[0,1]=1
    await driver(dut, tx)
    sb.check(tx, await monitor(dut, cfg), label="basic")


@cocotb.test()
async def test_groupsum_all_ones(dut: HierarchyObject) -> None:
    """4'b1111 -> group0=2, group1=2."""
    cfg = GroupSumConfig()
    sw = GroupSumSWModel(cfg)
    sb = Scoreboard(sw, cfg)

    await init_dut(dut)
    tx = GroupSumTx(bits=[1, 1, 1, 1])
    await driver(dut, tx)
    sb.check(tx, await monitor(dut, cfg), label="all_ones")


@cocotb.test()
async def test_groupsum_all_zeros(dut: HierarchyObject) -> None:
    """4'b0000 -> group0=0, group1=0."""
    cfg = GroupSumConfig()
    sw = GroupSumSWModel(cfg)
    sb = Scoreboard(sw, cfg)

    await init_dut(dut)
    tx = GroupSumTx(bits=[0, 0, 0, 0])
    await driver(dut, tx)
    sb.check(tx, await monitor(dut, cfg), label="all_zeros")


@cocotb.test()
async def test_groupsum_valid_ready(dut: HierarchyObject) -> None:
    """Valid/ready pass-through: output valid tracks input valid combinationally."""
    await init_dut(dut)
    dut.data_in_0.value = 0
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    await Timer(1, units="ns")
    assert int(dut.data_out_0_valid.value) == 1, "data_out_0_valid should be 1"
    assert int(dut.data_in_0_ready.value) == 1, "data_in_0_ready should be 1"


@cocotb.test()
async def test_groupsum_exhaustive(dut: HierarchyObject) -> None:
    """Exhaustive: all 16 4-bit input patterns vs PyTorch GroupSum reference."""
    cfg = GroupSumConfig()
    sw = GroupSumSWModel(cfg)
    sb = Scoreboard(sw, cfg)

    await init_dut(dut)
    for val in range(1 << cfg.input_size):
        bits = [(val >> i) & 1 for i in range(cfg.input_size)]
        tx = GroupSumTx(bits=bits)
        await driver(dut, tx)
        sb.check(tx, await monitor(dut, cfg), label=f"exhaust[{val:04b}]")

    cocotb.log.info(f"[SCOREBOARD] total: {sb.passed} passed, {sb.failed} failed")

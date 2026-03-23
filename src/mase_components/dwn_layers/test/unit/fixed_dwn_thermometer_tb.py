"""Cocotb testbench for fixed_dwn_thermometer"""

import random
from dataclasses import dataclass, field
from typing import List

import cocotb
from cocotb.handle import HierarchyObject
from cocotb.triggers import Timer
import torch


# Config

@dataclass
class ThermometerConfig:
    num_features: int = 2    # NUM_FEATURES
    feature_width: int = 8   # FEATURE_WIDTH (bits per feature)
    num_thresholds: int = 2  # NUM_THRESHOLDS
    # thresholds[f][t]: per-feature sorted threshold list
    thresholds: List[List[int]] = field(default_factory=lambda: [[64, 128], [32, 96]])

    def pack_thresholds(self) -> int:
        """Pack into RTL THRESHOLDS parameter: THRESHOLDS[(f*T+t)*W +: W]."""
        packed = 0
        mask = (1 << self.feature_width) - 1
        for f, row in enumerate(self.thresholds):
            for t, val in enumerate(row):
                bit_offset = (f * self.num_thresholds + t) * self.feature_width
                packed |= (val & mask) << bit_offset
        return packed


# Transaction (sequence item)

@dataclass
class ThermometerTx:
    features: List[int]  # unsigned integer feature values, length == num_features

    def to_rtl_input(self, cfg: ThermometerConfig) -> int:
        """Pack features into data_in_0: feature f at bits [f*W +: W]."""
        mask = (1 << cfg.feature_width) - 1
        val = 0
        for f, v in enumerate(self.features):
            val |= (v & mask) << (f * cfg.feature_width)
        return val


# SW Golden Model — PyTorch tensor comparisons, >= semantics matching RTL

class ThermometerSWModel:
    """
    Reference model using PyTorch tensor operations.

    Computes output_bit[f*T + t] = int(features[f] >= thresholds[f][t])
    using PyTorch tensors, matching the RTL's unsigned >= comparison.

    The chop.nn.dwn.DistributiveThermometer uses strict > internally, so
    this model reimplements the comparison directly with >= to match the RTL.
    """

    def __init__(self, cfg: ThermometerConfig):
        self.cfg = cfg
        # thresh shape: (F, T) int32
        self.thresh = torch.tensor(cfg.thresholds, dtype=torch.int32)

    def predict(self, tx: ThermometerTx) -> int:
        """Return packed output bits as integer (bit [f*T+t] = features[f] >= thresh[f][t])."""
        # x: (F, 1) — broadcast against thresh (F, T)
        x = torch.tensor(tx.features, dtype=torch.int32).unsqueeze(-1)
        out = (x >= self.thresh).int()   # (F, T); matches RTL >=
        bits = out.flatten().tolist()    # [b_{0,0}, b_{0,1}, b_{1,0}, b_{1,1}, ...]
        return sum(b << i for i, b in enumerate(bits))

    def predict_bits(self, tx: ThermometerTx) -> List[int]:
        """Return flat list of output bits for debug logging."""
        x = torch.tensor(tx.features, dtype=torch.int32).unsqueeze(-1)
        return (x >= self.thresh).int().flatten().tolist()


# Scoreboard

class Scoreboard:
    def __init__(self, sw: ThermometerSWModel):
        self.sw = sw
        self.passed = 0
        self.failed = 0

    def check(self, tx: ThermometerTx, rtl_out: int, label: str = "") -> None:
        sw_out = self.sw.predict(tx)
        tag = f"[SCOREBOARD]{' ' + label if label else ''}"
        if rtl_out == sw_out:
            cocotb.log.info(
                f"{tag} PASS  features={tx.features} "
                f"-> out={bin(rtl_out)} (0x{rtl_out:X})"
            )
            self.passed += 1
        else:
            self.failed += 1
            sw_bits = self.sw.predict_bits(tx)
            assert False, (
                f"{tag} features={tx.features}: "
                f"RTL={bin(rtl_out)}, SW={bin(sw_out)}, SW_bits={sw_bits}"
            )


# Infrastructure

async def init_dut(dut: HierarchyObject) -> None:
    dut.data_in_0_valid.value = 0
    dut.data_out_0_ready.value = 1
    await Timer(1, units="ns")


async def driver(
    dut: HierarchyObject, tx: ThermometerTx, cfg: ThermometerConfig
) -> None:
    dut.data_in_0.value = tx.to_rtl_input(cfg)
    dut.data_in_0_valid.value = 1
    dut.data_out_0_ready.value = 1
    await Timer(1, units="ns")
    dut.data_in_0_valid.value = 0


async def monitor(dut: HierarchyObject) -> int:
    """Capture packed data_out_0."""
    return int(dut.data_out_0.value)


# Tests (sequences)

@cocotb.test()
async def test_therm_basic(dut: HierarchyObject) -> None:
    """feature0=100, feature1=50 -> 0b0101 (SW >= model agrees)."""
    cfg = ThermometerConfig()
    sw = ThermometerSWModel(cfg)
    sb = Scoreboard(sw)

    await init_dut(dut)
    # f0=100: 100>=64=1, 100>=128=0 → bits[1:0]=0b01
    # f1=50:  50>=32=1,  50>=96=0  → bits[3:2]=0b01
    # packed  = 0b0101
    tx = ThermometerTx(features=[100, 50])
    await driver(dut, tx, cfg)
    sb.check(tx, await monitor(dut), label="basic")


@cocotb.test()
async def test_therm_below_all(dut: HierarchyObject) -> None:
    """Both features below all thresholds -> 0b0000."""
    cfg = ThermometerConfig()
    sw = ThermometerSWModel(cfg)
    sb = Scoreboard(sw)

    await init_dut(dut)
    tx = ThermometerTx(features=[10, 5])
    await driver(dut, tx, cfg)
    sb.check(tx, await monitor(dut), label="below_all")


@cocotb.test()
async def test_therm_above_all(dut: HierarchyObject) -> None:
    """Both features above all thresholds -> 0b1111."""
    cfg = ThermometerConfig()
    sw = ThermometerSWModel(cfg)
    sb = Scoreboard(sw)

    await init_dut(dut)
    tx = ThermometerTx(features=[200, 200])
    await driver(dut, tx, cfg)
    sb.check(tx, await monitor(dut), label="above_all")


@cocotb.test()
async def test_therm_exact_threshold(dut: HierarchyObject) -> None:
    """Boundary: feature == threshold.  RTL uses >=, so bit must be 1."""
    cfg = ThermometerConfig()
    sw = ThermometerSWModel(cfg)   # also uses >=, so agrees with RTL at boundary
    sb = Scoreboard(sw)

    await init_dut(dut)
    # feature0=64 == thresh[0][0]=64  → bit0=1
    # feature1=32 == thresh[1][0]=32  → bit2=1
    tx = ThermometerTx(features=[64, 32])
    await driver(dut, tx, cfg)
    sb.check(tx, await monitor(dut), label="exact_threshold")


@cocotb.test()
async def test_therm_random(dut: HierarchyObject) -> None:
    """32 random 8-bit feature pairs cross-checked against PyTorch >= reference."""
    cfg = ThermometerConfig()
    sw = ThermometerSWModel(cfg)
    sb = Scoreboard(sw)

    random.seed(42)
    await init_dut(dut)
    for i in range(32):
        features = [random.randint(0, 255) for _ in range(cfg.num_features)]
        tx = ThermometerTx(features=features)
        await driver(dut, tx, cfg)
        sb.check(tx, await monitor(dut), label=f"rand[{i}]")

    cocotb.log.info(f"[SCOREBOARD] total: {sb.passed} passed, {sb.failed} failed")

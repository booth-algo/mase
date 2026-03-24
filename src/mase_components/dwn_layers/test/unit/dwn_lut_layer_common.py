"""Shared classes for DWN LUT layer cocotb testbenches."""

from dataclasses import dataclass
from typing import List

import cocotb
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
        """Pack bits into RTL integer (bit 0 -> LSB)."""
        val = 0
        for i, b in enumerate(self.bits):
            val |= (b & 1) << i
        return val

    def to_torch(self) -> torch.Tensor:
        """Return int32 tensor (num_inputs,)."""
        return torch.tensor(self.bits, dtype=torch.int32)


# SW Golden Model -- PyTorch tensor LUT lookup

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

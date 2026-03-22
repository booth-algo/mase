"""UVM-style cocotb testbench for DWN MNIST full simulation.

Architecture mirrors UVM:
  - Transaction   : single test vector (thermo_bits → expected_hw_bits + label)
  - Sequencer     : generates Transaction objects from JSON config
  - Driver        : drives DUT inputs asynchronously
  - Monitor       : captures DUT outputs asynchronously
  - Scoreboard    : compares actual vs expected, tracks per-class stats
  - CoverageCollector : per-class coverage bins
  - TestEnv       : wires everything together

Config JSON (DWN_MNIST_UVM_CONFIG env var):
  {
    "num_inputs":  int,    # thermometer width (2352 for MNIST n2)
    "num_outputs": int,    # last LUT layer output width (1000 for MNIST n2)
    "num_classes": int,    # number of output classes (10 for MNIST)
    "transactions": [
      {
        "thermo_packed":     int,  # packed thermometer input bits
        "hw_output_packed":  int,  # expected packed LUT output from torch
        "label":             int,  # MNIST ground-truth class
        "sw_pred":           int   # torch model predicted class (for reference)
      }, ...
    ]
  }
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import cocotb
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.triggers import ClockCycles, RisingEdge


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """One MNIST inference test vector."""
    idx: int               # transaction index
    thermo_packed: int     # packed thermometer input bits
    hw_output_packed: int  # expected packed LUT output (from torch)
    label: int             # ground-truth MNIST class
    sw_pred: int           # torch model predicted class (for reference)


# ---------------------------------------------------------------------------
# Sequencer
# ---------------------------------------------------------------------------

class Sequencer:
    """Loads transactions from config and feeds them to the Driver queue."""

    def __init__(self, config: dict, driver_queue: Queue):
        self._transactions: List[Transaction] = []
        for i, t in enumerate(config["transactions"]):
            self._transactions.append(Transaction(
                idx=i,
                thermo_packed=t["thermo_packed"],
                hw_output_packed=t["hw_output_packed"],
                label=t["label"],
                sw_pred=t["sw_pred"],
            ))
        self._driver_queue = driver_queue

    async def run(self):
        for txn in self._transactions:
            await self._driver_queue.put(txn)
        # Sentinel to signal end
        await self._driver_queue.put(None)

    def __len__(self):
        return len(self._transactions)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Driver:
    """Drives thermometer-encoded inputs into the DUT."""

    def __init__(self, dut, driver_queue: Queue, monitor_queue: Queue):
        self._dut = dut
        self._driver_queue = driver_queue
        self._monitor_queue = monitor_queue  # pass transaction to monitor for scoreboard

    async def run(self):
        dut = self._dut
        while True:
            txn: Optional[Transaction] = await self._driver_queue.get()
            if txn is None:
                # Sentinel: signal monitor that all transactions are sent
                await self._monitor_queue.put(None)
                break
            dut.data_in_0.value = txn.thermo_packed
            dut.data_in_0_valid.value = 1
            dut.data_out_0_ready.value = 1
            await RisingEdge(dut.clk)
            await ClockCycles(dut.clk, 1)   # combinational settle
            dut.data_in_0_valid.value = 0
            # Forward transaction metadata to monitor
            await self._monitor_queue.put(txn)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

@dataclass
class ObservedOutput:
    txn: Transaction
    rtl_output_packed: int


class Monitor:
    """Captures DUT output for each driven transaction."""

    def __init__(self, dut, monitor_queue: Queue, scoreboard_queue: Queue):
        self._dut = dut
        self._monitor_queue = monitor_queue
        self._scoreboard_queue = scoreboard_queue

    async def run(self):
        dut = self._dut
        while True:
            txn: Optional[Transaction] = await self._monitor_queue.get()
            if txn is None:
                await self._scoreboard_queue.put(None)
                break
            # Sample output (combinational — valid after ClockCycles in driver)
            rtl_out = int(dut.data_out_0.value)
            obs = ObservedOutput(txn=txn, rtl_output_packed=rtl_out)
            await self._scoreboard_queue.put(obs)


# ---------------------------------------------------------------------------
# Coverage Collector
# ---------------------------------------------------------------------------

class CoverageCollector:
    """Functional coverage: tracks which MNIST classes were tested and passed."""

    def __init__(self, num_classes: int = 10):
        self._bins = {c: {"seen": 0, "pass": 0} for c in range(num_classes)}

    def record(self, label: int, passed: bool):
        self._bins[label]["seen"] += 1
        if passed:
            self._bins[label]["pass"] += 1

    def report(self):
        cocotb.log.info("  Coverage bins (per MNIST class):")
        all_covered = True
        for c, stats in self._bins.items():
            seen = stats["seen"]
            pct = 100 * stats["pass"] / max(seen, 1)
            covered = seen >= 5  # at least 5 samples per class
            all_covered = all_covered and covered
            cocotb.log.info(
                f"    Class {c}: {seen:4d} samples, {pct:5.1f}% RTL match "
                f"{'[OK]' if covered else '[UNDER-COVERED]'}"
            )
        return all_covered


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

class Scoreboard:
    """Compares RTL output vs SW golden model; tracks per-class stats."""

    def __init__(self, num_outputs: int, num_classes: int = 10):
        self._num_outputs = num_outputs
        self._num_classes = num_classes
        self._pass = 0
        self._fail = 0
        self._per_class_pass = [0] * num_classes
        self._per_class_fail = [0] * num_classes
        self._mismatches = []

    async def run(self, scoreboard_queue: Queue, coverage: CoverageCollector):
        while True:
            obs: Optional[ObservedOutput] = await scoreboard_queue.get()
            if obs is None:
                break

            txn = obs.txn
            expected = txn.hw_output_packed
            actual = obs.rtl_output_packed
            match = (actual == expected)
            label = txn.label

            coverage.record(label, match)

            if match:
                self._pass += 1
                self._per_class_pass[label] += 1
            else:
                self._fail += 1
                self._per_class_fail[label] += 1
                if len(self._mismatches) < 20:
                    self._mismatches.append({
                        "idx": txn.idx,
                        "label": label,
                        "sw_pred": txn.sw_pred,
                        "expected": hex(expected),
                        "actual":   hex(actual),
                        "xor":      hex(expected ^ actual),
                        "diffbits": bin(expected ^ actual).count("1"),
                    })

    def report(self):
        total = self._pass + self._fail
        sep = "=" * 62
        cocotb.log.info(
            f"\n{sep}\n"
            f"  DWN MNIST RTL vs Torch Golden Reference — Equivalence Report\n"
            f"{sep}\n"
            f"  Total transactions : {total}\n"
            f"  PASS (RTL==SW)     : {self._pass}  ({100*self._pass/max(total,1):.1f}%)\n"
            f"  FAIL (RTL!=SW)     : {self._fail}\n"
        )
        cocotb.log.info("  Per-class breakdown:")
        for c in range(self._num_classes):
            p = self._per_class_pass[c]
            f_ = self._per_class_fail[c]
            tot = p + f_
            acc = 100 * p / max(tot, 1)
            cocotb.log.info(f"    Class {c}: {p:4d}/{tot:4d} pass  ({acc:5.1f}%)")

        if self._mismatches:
            cocotb.log.warning(f"\n  First {len(self._mismatches)} mismatches:")
            for m in self._mismatches:
                cocotb.log.warning(
                    f"    txn#{m['idx']:5d}  label={m['label']}  sw_pred={m['sw_pred']}  "
                    f"diffbits={m['diffbits']}  xor={m['xor']}"
                )
        cocotb.log.info(sep)
        return self._fail == 0


# ---------------------------------------------------------------------------
# TestEnv  (top-level environment — wires all UVM components)
# ---------------------------------------------------------------------------

class TestEnv:
    """Top-level test environment: Sequencer → Driver → Monitor → Scoreboard."""

    def __init__(self, dut, config: dict):
        self._config = config
        self._dut = dut
        self._driver_queue: Queue = Queue()
        self._monitor_queue: Queue = Queue()
        self._scoreboard_queue: Queue = Queue()

        self._sequencer    = Sequencer(config, self._driver_queue)
        self._driver       = Driver(dut, self._driver_queue, self._monitor_queue)
        self._monitor      = Monitor(dut, self._monitor_queue, self._scoreboard_queue)
        self._coverage     = CoverageCollector(num_classes=config.get("num_classes", 10))
        self._scoreboard   = Scoreboard(
            num_outputs=config["num_outputs"],
            num_classes=config.get("num_classes", 10),
        )

    async def run(self):
        dut = self._dut

        # Clock and reset
        await cocotb.start(Clock(dut.clk, 3.2, units="ns").start())
        dut.rst.value = 1
        dut.data_in_0_valid.value = 0
        dut.data_out_0_ready.value = 1
        await ClockCycles(dut.clk, 4)
        dut.rst.value = 0
        await RisingEdge(dut.clk)

        # Start all UVM components concurrently
        seq_task = await cocotb.start(self._sequencer.run())
        drv_task = await cocotb.start(self._driver.run())
        mon_task = await cocotb.start(self._monitor.run())
        sb_task  = await cocotb.start(
            self._scoreboard.run(self._scoreboard_queue, self._coverage)
        )

        # Wait for pipeline to drain
        await seq_task
        await drv_task
        await mon_task
        await sb_task

        # Final reports
        self._coverage.report()
        passed = self._scoreboard.report()
        return passed


# ---------------------------------------------------------------------------
# cocotb test entry point
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = os.environ.get("DWN_MNIST_UVM_CONFIG", "")
    if not config_path:
        raise RuntimeError("DWN_MNIST_UVM_CONFIG environment variable is not set")
    with open(config_path) as f:
        return json.load(f)


@cocotb.test()
async def test_dwn_mnist_uvm(dut):
    """
    UVM-style full simulation of dwn_top RTL against torch DWN golden reference.

    Drives real MNIST test vectors (thermometer-encoded) through the emitted
    dwn_top.sv RTL and verifies that the RTL output matches the torch DWN
    hardware-core output bit-for-bit.

    Components:
      Sequencer     — iterates MNIST transactions from JSON config
      Driver        — drives DUT data_in_0 with packed thermometer bits
      Monitor       — samples DUT data_out_0 after each transaction
      Scoreboard    — compares RTL output vs torch golden expected output
      Coverage      — per-MNIST-class coverage bins
    """
    config = _load_config()

    n_txn = len(config["transactions"])
    cocotb.log.info(
        f"[DWN MNIST UVM] Starting simulation: "
        f"{n_txn} transactions, "
        f"num_inputs={config['num_inputs']}, "
        f"num_outputs={config['num_outputs']}, "
        f"num_classes={config.get('num_classes', 10)}"
    )

    env = TestEnv(dut, config)
    all_passed = await env.run()

    assert all_passed, (
        "RTL vs torch golden reference MISMATCH detected — "
        "check scoreboard report above for details"
    )

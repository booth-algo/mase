"""UVM-style cocotb testbench for DWN paper-scope full simulation.

Architecture mirrors UVM:
  - Transaction   : single test vector (thermo_packed → expected_scores + label)
  - Sequencer     : generates Transaction objects from JSON config
  - Driver        : drives DUT inputs clocked (4-cycle pipeline)
  - Monitor       : captures DUT outputs after PIPELINE_DEPTH rising edges
  - Scoreboard    : compares RTL scores vs SW golden model, tracks per-class stats
  - CoverageCollector : per-class coverage bins
  - TestEnv       : wires everything together

Config JSON (DWN_PAPER_SCOPE_UVM_CONFIG env var):
  {
    "num_inputs":  int,    # thermometer width (2352 for baseline_n6)
    "num_classes": int,    # number of output classes (10)
    "transactions": [
      {
        "thermo_packed":    int,       # packed thermometer input bits
        "expected_scores":  [int, ...], # list of 10 raw group counts (0-100 each)
        "label":            int,       # ground-truth class
        "sw_pred":          int        # SW model predicted class (for reference)
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


# Total pipeline latency: 2 cycles (dwn_top_clocked) + 2 cycles (groupsum_pipelined)
PIPELINE_DEPTH = 4


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """One inference test vector for the paper-scope clocked pipeline."""
    idx: int
    thermo_packed: int
    expected_scores: list   # list of 10 ints (raw group counts, 0-100 each)
    label: int
    sw_pred: int


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
                expected_scores=t["expected_scores"],
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
    """Drives thermometer-encoded inputs into the clocked DUT."""

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
            # Wait PIPELINE_DEPTH cycles so driver and monitor stay in lockstep:
            # txn enters FF1 on edge E1; output is committed at E(PIPELINE_DEPTH+1) NBA.
            # Monitor will wait 1 more edge to cross that NBA boundary.
            await ClockCycles(dut.clk, PIPELINE_DEPTH)
            # Forward transaction metadata to monitor
            await self._monitor_queue.put(txn)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

@dataclass
class ObservedOutput:
    txn: Transaction
    rtl_scores: list  # list of 10 ints read from RTL flat bus


class Monitor:
    """Captures DUT output for each driven transaction after PIPELINE_DEPTH cycles."""

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
            # Driver already waited PIPELINE_DEPTH cycles after presenting txn.
            # At the point the monitor receives the txn, the pipeline output is
            # committed at the CURRENT clock's NBA phase.  We wait 1 more edge
            # so that cocotb reads in the Active phase AFTER that NBA commits.
            await ClockCycles(dut.clk, 1)
            flat_val = int(dut.data_out_0_flat.value)
            rtl_scores = [(flat_val >> (i * 8)) & 0xFF for i in range(10)]
            obs = ObservedOutput(txn=txn, rtl_scores=rtl_scores)
            await self._scoreboard_queue.put(obs)


# ---------------------------------------------------------------------------
# Coverage Collector
# ---------------------------------------------------------------------------

class CoverageCollector:
    """Functional coverage: tracks which classes were tested and passed."""

    def __init__(self, num_classes: int = 10):
        self._bins = {c: {"seen": 0, "pass": 0} for c in range(num_classes)}

    def record(self, label: int, passed: bool):
        self._bins[label]["seen"] += 1
        if passed:
            self._bins[label]["pass"] += 1

    def report(self):
        cocotb.log.info("  Coverage bins (per class):")
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
    """Compares RTL scores vs SW golden model; tracks per-class stats."""

    def __init__(self, num_classes: int = 10):
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
            expected = txn.expected_scores
            actual = obs.rtl_scores
            # Exact element-wise match across all 10 scores
            match = (actual == expected)
            label = txn.label

            # Per-class accuracy: does argmax(rtl_scores) == label?
            rtl_pred = int(max(range(self._num_classes), key=lambda c: actual[c]))
            coverage.record(label, match)

            if match:
                self._pass += 1
                self._per_class_pass[label] += 1
            else:
                self._fail += 1
                self._per_class_fail[label] += 1
                if len(self._mismatches) < 20:
                    diffs = [
                        (i, expected[i], actual[i])
                        for i in range(self._num_classes)
                        if expected[i] != actual[i]
                    ]
                    self._mismatches.append({
                        "idx":      txn.idx,
                        "label":    label,
                        "sw_pred":  txn.sw_pred,
                        "rtl_pred": rtl_pred,
                        "expected": expected,
                        "actual":   actual,
                        "diffs":    diffs,
                    })

    def report(self):
        total = self._pass + self._fail
        sep = "=" * 66
        cocotb.log.info(
            f"\n{sep}\n"
            f"  DWN Paper-Scope RTL vs SW Golden Reference — Equivalence Report\n"
            f"{sep}\n"
            f"  Total transactions : {total}\n"
            f"  PASS (RTL==SW)     : {self._pass}  ({100*self._pass/max(total,1):.1f}%)\n"
            f"  FAIL (RTL!=SW)     : {self._fail}\n"
        )
        cocotb.log.info("  Per-class breakdown (by ground-truth label):")
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
                    f"    txn#{m['idx']:5d}  label={m['label']}  "
                    f"sw_pred={m['sw_pred']}  rtl_pred={m['rtl_pred']}  "
                    f"score_diffs={m['diffs']}"
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

        self._sequencer  = Sequencer(config, self._driver_queue)
        self._driver     = Driver(dut, self._driver_queue, self._monitor_queue)
        self._monitor    = Monitor(dut, self._monitor_queue, self._scoreboard_queue)
        self._coverage   = CoverageCollector(num_classes=config.get("num_classes", 10))
        self._scoreboard = Scoreboard(num_classes=config.get("num_classes", 10))

    async def run(self):
        dut = self._dut

        # Clock and reset
        await cocotb.start(Clock(dut.clk, 3.2, units="ns").start())
        dut.rst.value = 1
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
    config_path = os.environ.get("DWN_PAPER_SCOPE_UVM_CONFIG", "")
    if not config_path:
        raise RuntimeError("DWN_PAPER_SCOPE_UVM_CONFIG environment variable is not set")
    with open(config_path) as f:
        return json.load(f)


@cocotb.test()
async def test_dwn_paper_scope_uvm(dut):
    """
    UVM-style full simulation of dwn_paper_scope_sim_wrapper RTL against
    SW golden reference (Python group_sum on sw_forward LUT outputs).

    Drives real MNIST test vectors (thermometer-encoded) through the clocked
    dwn_top_paper_scope pipeline (4-cycle latency) and verifies that the RTL
    class scores exactly match the Python golden group_sum counts.

    Components:
      Sequencer     — iterates transactions from JSON config
      Driver        — drives DUT data_in_0 on each rising edge
      Monitor       — captures DUT data_out_0_flat after PIPELINE_DEPTH=4 cycles
      Scoreboard    — compares RTL scores vs SW expected_scores element-wise
      Coverage      — per-class coverage bins
    """
    config = _load_config()

    n_txn = len(config["transactions"])
    cocotb.log.info(
        f"[DWN Paper-Scope UVM] Starting simulation: "
        f"{n_txn} transactions, "
        f"num_inputs={config['num_inputs']}, "
        f"num_classes={config.get('num_classes', 10)}, "
        f"pipeline_depth={PIPELINE_DEPTH}"
    )

    env = TestEnv(dut, config)
    all_passed = await env.run()

    assert all_passed, (
        "RTL vs SW golden reference MISMATCH detected — "
        "check scoreboard report above for details"
    )

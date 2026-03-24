"""Cocotb testbench for structural_dwn_lut_neuron.

Exhaustively tests all 2^LUT_N input patterns against the expected
LUT_CONTENTS truth table. For LUT_N=2, LUT_CONTENTS=4'b1010:
    addr=0 -> 0, addr=1 -> 1, addr=2 -> 0, addr=3 -> 1
which implements out = input[0] (identity of bit 0).
"""

import cocotb
from cocotb.handle import HierarchyObject


@cocotb.test()
async def test_neuron_exhaustive(dut: HierarchyObject) -> None:
    """Drive all 2^LUT_N input patterns and check output matches LUT_CONTENTS."""
    # Read parameters from the DUT generics.
    lut_n = int(dut.LUT_N.value)
    # LUT_CONTENTS is passed as a parameter; extract its integer value.
    lut_contents = int(dut.LUT_CONTENTS.value)
    num_patterns = 1 << lut_n

    cocotb.log.info(
        f"Testing structural_dwn_lut_neuron: LUT_N={lut_n}, "
        f"LUT_CONTENTS=0b{lut_contents:0{num_patterns}b}"
    )

    passed = 0
    failed = 0

    for addr in range(num_patterns):
        # Drive input
        dut.data_in_0.value = addr

        # Allow combinational propagation (no clock needed — purely combinational)
        await cocotb.triggers.Timer(1, units="ns")

        # Read output
        rtl_out = int(dut.data_out_0.value)
        expected = (lut_contents >> addr) & 1

        if rtl_out == expected:
            cocotb.log.info(
                f"  addr={addr:0{lut_n}b} -> out={rtl_out} (expected {expected}) PASS"
            )
            passed += 1
        else:
            failed += 1
            assert False, (
                f"addr={addr:0{lut_n}b}: RTL={rtl_out}, expected={expected} "
                f"(LUT_CONTENTS=0b{lut_contents:0{num_patterns}b})"
            )

    cocotb.log.info(f"[NEURON TB] total: {passed} passed, {failed} failed")

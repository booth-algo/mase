# Vivado full-implementation synthesis for DWN paper replication.
#
# This script targets Flow_PerfOptimized_high to match the synthesis strategy
# used in Bacellar et al. (arXiv:2410.11112) Table 2 OOC results. Unlike
# synth_dwn.tcl (synth-only), this script runs the full implementation flow
# (opt → place → phys_opt → route) with aggressive performance directives.
#
# Usage:
#   vivado -mode batch -source scripts/synth_paper_match.tcl \
#          -tclargs <rtl_dir> <results_dir> [<part> [<top_module> [<clk_period_ns>]]]
#
# Arguments:
#   rtl_dir        Directory containing .sv RTL files
#   results_dir    Directory to write utilization/timing/power reports
#   part           (optional) Xilinx part — default xcvu9p-flgb2104-2-i
#   top_module     (optional) Top module name — default dwn_top_paper_scope
#   clk_period_ns  (optional) Clock period in ns — default 1.21 (827 MHz)
#
# Example — paper-scope MNIST lg (LUT layers + pipelined GroupSum):
#   vivado -mode batch -source scripts/synth_paper_match.tcl \
#          -tclargs mase_output/dwn/baseline_n6_rtl/hardware/rtl \
#                   mase_output/dwn/baseline_n6_paper_results \
#                   xcvu9p-flgb2104-2-i dwn_top_paper_scope 1.21

# -----------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------
set rtl_dir       [lindex $argv 0]
set results_dir   [lindex $argv 1]
set part          [expr { [llength $argv] > 2 ? [lindex $argv 2] : "xcvu9p-flgb2104-2-i" }]
set top_module    [expr { [llength $argv] > 3 ? [lindex $argv 3] : "dwn_top_paper_scope" }]
set clk_period_ns [expr { [llength $argv] > 4 ? [lindex $argv 4] : "1.21" }]

file mkdir $results_dir

puts "================================================================"
puts "DWN Paper-Match Synthesis (Flow_PerfOptimized_high)"
puts "  RTL dir    : $rtl_dir"
puts "  Results dir: $results_dir"
puts "  Part       : $part"
puts "  Top module : $top_module"
puts "  Clk period : $clk_period_ns ns  ([format %.1f [expr {1000.0 / $clk_period_ns}]] MHz target)"
puts "================================================================"

# -----------------------------------------------------------------------
# Create in-memory project and read sources
# -----------------------------------------------------------------------
create_project -in_memory -part $part

set sv_files [glob -nocomplain "$rtl_dir/*.sv"]
if { [llength $sv_files] == 0 } {
    puts "ERROR: No .sv files found in $rtl_dir"
    exit 1
}
read_verilog -sv $sv_files
set _fnames {}
foreach _f $sv_files { lappend _fnames [file tail $_f] }
puts "Read [llength $sv_files] source file(s): [join $_fnames {, }]"

set_property top $top_module [current_fileset]

# -----------------------------------------------------------------------
# Synthesise — PerformanceOptimized + flatten_hierarchy rebuilt
# -----------------------------------------------------------------------
synth_design \
    -top $top_module \
    -part $part \
    -mode out_of_context \
    -flatten_hierarchy rebuilt \
    -directive PerformanceOptimized

# Clock constraint (applied after synth_design opens the netlist)
create_clock -period $clk_period_ns -name clk [get_ports clk]

# -----------------------------------------------------------------------
# Implementation — Flow_PerfOptimized_high directives
# -----------------------------------------------------------------------
opt_design      -directive ExploreWithRemap
place_design    -directive ExtraPostPlacementOpt
phys_opt_design -directive AggressiveExplore
route_design    -directive AggressiveExplore

# -----------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------
report_utilization    -file "$results_dir/utilization.rpt"
report_timing_summary -max_paths 10 -file "$results_dir/timing.rpt"
report_power          -file "$results_dir/power.rpt"

# Print key numbers to stdout
puts "\n=== Utilization summary ==="
report_utilization -return_string

puts "\n=== Timing ==="
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "WNS: $wns ns"

# Fmax calculation:
#   If timing met (WNS >= 0): Fmax = 1000 / clk_period_ns
#   If timing violated (WNS < 0): Fmax = 1000 / (clk_period_ns - WNS)
if { $wns >= 0 } {
    set fmax [expr { 1000.0 / $clk_period_ns }]
    puts [format "Fmax: %.1f MHz  (timing met, WNS=+%.3f ns)" $fmax $wns]
} else {
    set fmax [expr { 1000.0 / ($clk_period_ns - $wns) }]
    puts [format "Fmax: %.1f MHz  (timing violated, WNS=%.3f ns)" $fmax $wns]
}

puts "\nDone. Reports written to $results_dir"
close_project
exit

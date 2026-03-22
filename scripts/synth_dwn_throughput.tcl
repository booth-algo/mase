# Vivado throughput-optimised synthesis for DWN — matches Flow_PerfOptimized_high.
#
# Usage:
#   vivado -mode batch -source synth_dwn_throughput.tcl \
#          -tclargs <rtl_dir> <results_dir> [<part> [<top_module> [<clk_period_ns>]]]
#
# Differences from synth_dwn.tcl:
#   - Implementation directives: AggressiveExplore (place/route) + phys_opt step
#   - Clock constraint applied BEFORE opt_design (ensures timing-driven placement)
#   - Matches Vivado "Flow_PerfOptimized_high" non-project strategy
#
# Paper comparison target (Bacellar et al. OOC Appendix D):
#   part=xcvu9p-flgb2104-2-i  top_module=full_pipeline_top_clocked  clk_period_ns=4.0

# -----------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------
set rtl_dir       [lindex $argv 0]
set results_dir   [lindex $argv 1]
set part          [expr { [llength $argv] > 2 ? [lindex $argv 2] : "xcvc1902-viva1596-3HP-e-S" }]
set top_module    [expr { [llength $argv] > 3 ? [lindex $argv 3] : "dwn_top" }]
set clk_period_ns [expr { [llength $argv] > 4 ? [lindex $argv 4] : "2.0" }]

file mkdir $results_dir

puts "================================================================"
puts "DWN Throughput Synthesis (Flow_PerfOptimized_high)"
puts "  RTL dir    : $rtl_dir"
puts "  Results dir: $results_dir"
puts "  Part       : $part"
puts "  Top module : $top_module"
puts "  Clk period : $clk_period_ns ns"
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
# Synthesise — PerformanceOptimized (same as area run; pruning happens here)
# -----------------------------------------------------------------------
synth_design \
    -top $top_module \
    -part $part \
    -mode out_of_context \
    -flatten_hierarchy rebuilt \
    -directive PerformanceOptimized

# Apply clock constraint for pipelined variants (after synth opens the design)
if { [string match "*_clocked" $top_module] } {
    create_clock -period $clk_period_ns -name clk [get_ports clk]
    puts "Clock constraint: $clk_period_ns ns ([expr {1000.0 / $clk_period_ns}] MHz target)"
}

# -----------------------------------------------------------------------
# Implementation — Flow_PerfOptimized_high strategy
# opt:     Explore     (more LUT combining, retiming)
# place:   ExtraPostPlacementOpt  (post-placement optimisation pass)
# phys:    AggressiveExplore      (timing-driven physical optimisation)
# route:   AggressiveExplore      (congestion + timing co-optimised routing)
# post-r:  AggressiveExplore      (post-route physical opt)
# -----------------------------------------------------------------------
opt_design   -directive Explore
place_design -directive ExtraPostPlacementOpt
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
phys_opt_design -directive AggressiveExplore

# -----------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------
report_utilization   -file "$results_dir/utilization.rpt"
report_timing_summary -max_paths 10 -file "$results_dir/timing.rpt"
report_power         -file "$results_dir/power.rpt"

# Print key numbers to stdout for quick inspection
puts "\n=== Utilization summary ==="
report_utilization -return_string

puts "\n=== Worst negative slack ==="
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "WNS: $wns ns"

puts "\nDone. Reports written to $results_dir"
close_project
exit

# Vivado out-of-context synthesis for DWN top-level.
#
# Usage:
#   vivado -mode batch -source synth_dwn.tcl \
#          -tclargs <rtl_dir> <results_dir> [<part> [<top_module> [<clk_period_ns>]]]
#
# Arguments:
#   rtl_dir        Directory containing dwn_top.sv and RTL dependencies
#   results_dir    Directory to write utilization/timing reports
#   part           (optional) Xilinx part string — default xcvc1902-viva1596-3HP-e-S
#   top_module     (optional) Top module name — default dwn_top
#   clk_period_ns  (optional) Clock period in ns for pipelined variant — default 2.0
#
# Paper comparison target (Bacellar et al. OOC Appendix D):
#   part=xcvu9p-flgb2104-2-i  clk_period_ns=4.0  (250 MHz)

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
puts "DWN Synthesis"
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
# Synthesise (out-of-context, performance-optimised — matches Jino et al.)
# -----------------------------------------------------------------------
synth_design \
    -top $top_module \
    -part $part \
    -mode out_of_context \
    -flatten_hierarchy rebuilt \
    -directive PerformanceOptimized

# Add clock constraint when synthesising the pipelined variant
# (must be called after synth_design opens the design)
if { [string match "*_clocked" $top_module] } {
    create_clock -period $clk_period_ns -name clk [get_ports clk]
}

# -----------------------------------------------------------------------
# Assess maximum frequency (AMD guide: iterate implementation until timing met)
# -----------------------------------------------------------------------
opt_design
place_design
route_design

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

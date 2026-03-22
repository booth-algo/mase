# Vivado out-of-context synthesis for DiffLogic top-level.
#
# Usage:
#   vivado -mode batch -source scripts/synth_difflogic.tcl \
#          -tclargs <rtl_dir> <results_dir> [<part>]
#
# Arguments:
#   rtl_dir     Directory containing difflogic_top.sv and RTL dependencies
#   results_dir Directory to write utilization/timing reports
#   part        (optional) Xilinx part string — default xcvc1902-viva1596-3HP-e-S
#
# Matching Jino et al. (DiffLogic+MASE, Imperial College) setup for comparability.

# -----------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------
set rtl_dir     [lindex $argv 0]
set results_dir [lindex $argv 1]
set part        [expr { [llength $argv] > 2 ? [lindex $argv 2] : "xcvc1902-viva1596-3HP-e-S" }]

file mkdir $results_dir

puts "================================================================"
puts "DiffLogic Synthesis"
puts "  RTL dir    : $rtl_dir"
puts "  Results dir: $results_dir"
puts "  Part       : $part"
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
puts "Read [llength $sv_files] source file(s): [join [lmap f $sv_files {file tail $f}] {, }]"

set_property top difflogic_top [current_fileset]

# -----------------------------------------------------------------------
# Synthesise (out-of-context, performance-optimised — matches Jino et al.)
# -----------------------------------------------------------------------
synth_design \
    -top difflogic_top \
    -part $part \
    -mode out_of_context \
    -flatten_hierarchy rebuilt \
    -directive PerformanceOptimized

# -----------------------------------------------------------------------
# Assess maximum frequency (place + route for timing analysis)
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

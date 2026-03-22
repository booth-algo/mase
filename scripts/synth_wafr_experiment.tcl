# synth_wafr_experiment.tcl
# Matches Mecik & Kumm (arXiv:2512.15251) synthesis methodology:
#   - xcvu9p-flgb2104-2-i (AMD Virtex UltraScale+)
#   - Out-of-context (OOC) synthesis
#   - Flow_PerfOptimized_high strategy
#   - 700 MHz target clock
#
# Usage:
#   vivado -mode batch -source synth_wafr_experiment.tcl \
#          -tclargs <rtl_dir> <results_dir> [top_module]

# Parse arguments
set rtl_dir    [lindex $argv 0]
set result_dir [lindex $argv 1]
set top_module [expr {[llength $argv] > 2 ? [lindex $argv 2] : "full_pipeline_top_clocked"}]

puts "============================================"
puts "WAFR Packing Experiment"
puts "RTL dir:    $rtl_dir"
puts "Results:    $result_dir"
puts "Top module: $top_module"
puts "Part:       xcvu9p-flgb2104-2-i"
puts "Clock:      700 MHz (1.428 ns)"
puts "Strategy:   Flow_PerfOptimized_high (OOC)"
puts "============================================"

# Create results directory
file mkdir $result_dir

# Read all SystemVerilog sources
foreach f [glob -directory $rtl_dir *.sv] {
    read_verilog -sv $f
    puts "Read: $f"
}

# Out-of-context synthesis on xcvu9p
set_property -dict {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS {-mode out_of_context}} [get_runs synth_1] -quiet
synth_design -top $top_module -part xcvu9p-flgb2104-2-i -mode out_of_context

# Create 700 MHz clock (1.428 ns period)
create_clock -period 1.428 -name clk [get_ports clk]

# Flow_PerfOptimized_high implementation directives
opt_design -directive Explore
place_design -directive ExtraPostPlacementOpt
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
phys_opt_design -directive AggressiveExplore

# Generate reports
report_utilization -file ${result_dir}/utilization.rpt
report_timing_summary -file ${result_dir}/timing.rpt
report_timing -nworst 10 -file ${result_dir}/timing_paths.rpt

# Extract and print key metrics
puts "\n============================================"
puts "RESULTS SUMMARY"
puts "============================================"

# Get LUT count
set lut_count [get_property STATS.LUT [current_design]]
puts "LUTs: $lut_count"

# Get FF count
set ff_count [get_property STATS.FF [current_design]]
puts "FFs:  $ff_count"

# Get WNS (Worst Negative Slack)
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1]]
puts "WNS:  $wns ns"

# Calculate Fmax
set period 1.428
set fmax [expr {1000.0 / ($period - $wns)}]
puts "Fmax: [format %.1f $fmax] MHz"

puts "============================================"
puts "Reports saved to: $result_dir"
puts "============================================"

# Save checkpoint
write_checkpoint -force ${result_dir}/post_route.dcp

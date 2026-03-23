/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
// Simulation wrapper for dwn_top_paper_scope.
// Packs the unpacked output array data_out_0[0:9] into a flat 80-bit bus
// for easy access in cocotb/Verilator (which doesn't directly expose unpacked arrays).
// data_out_0_flat[7:0]   = score[0] (class 0)
// data_out_0_flat[15:8]  = score[1] (class 1)
// ...
// data_out_0_flat[79:72] = score[9] (class 9)
module dwn_paper_scope_sim_wrapper (
    input  wire           clk,
    input  wire           rst,
    input  wire  [2351:0] data_in_0,
    output logic [79:0]   data_out_0_flat
);

    logic [7:0] scores [0:9];

    dwn_top_paper_scope dut (
        .clk        (clk),
        .rst        (rst),
        .data_in_0  (data_in_0),
        .data_out_0 (scores)
    );

    genvar i;
    generate
        for (i = 0; i < 10; i = i + 1) begin : gen_pack
            assign data_out_0_flat[i*8 +: 8] = scores[i];
        end
    endgenerate

endmodule

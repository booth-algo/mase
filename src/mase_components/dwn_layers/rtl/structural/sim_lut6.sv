// Behavioral simulation model for Xilinx LUT6 primitive.
// Used only for Verilator/cocotb simulation of structural DWN variants.
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
module LUT6 #(
    parameter [63:0] INIT = 64'h0
) (
    output wire O,
    input  wire I0, I1, I2, I3, I4, I5
);
    wire [5:0] addr = {I5, I4, I3, I2, I1, I0};
    assign O = INIT[addr];
endmodule

module fixed_dwn_lut_neuron #(
    parameter LUT_N                         = 6,
    parameter [(2**LUT_N)-1:0] LUT_CONTENTS = {(2**LUT_N){1'b0}}
) (
    input  wire [LUT_N-1:0]     data_in_0,
    output wire                 data_out_0
);

    assign data_out_0   = LUT_CONTENTS[data_in_0];

endmodule

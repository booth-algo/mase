// structural_dwn_lut_neuron.sv
// Uses explicit Xilinx LUT6 primitive to prevent LUT combining/packing.
// Each neuron occupies exactly one LUT6 site.
// Only supports LUT_N <= 6.

module structural_dwn_lut_neuron #(
    parameter LUT_N                         = 6,
    parameter [(2**LUT_N)-1:0] LUT_CONTENTS = {(2**LUT_N){1'b0}}
) (
    input  wire [LUT_N-1:0] data_in_0,
    output wire              data_out_0
);

generate
    if (LUT_N == 6) begin : gen_lut6
        (* DONT_TOUCH = "TRUE" *)
        LUT6 #(
            .INIT(LUT_CONTENTS)
        ) lut_inst (
            .O  (data_out_0),
            .I0 (data_in_0[0]),
            .I1 (data_in_0[1]),
            .I2 (data_in_0[2]),
            .I3 (data_in_0[3]),
            .I4 (data_in_0[4]),
            .I5 (data_in_0[5])
        );
    end
    else if (LUT_N <= 5) begin : gen_lut5_in_lut6
        localparam REPLICAS = 2**(6 - LUT_N);
        localparam [63:0] INIT_64 = {REPLICAS{LUT_CONTENTS}};

        wire [5:0] padded_in;
        assign padded_in[LUT_N-1:0] = data_in_0;
        if (LUT_N < 6)
            assign padded_in[5:LUT_N] = {(6-LUT_N){1'b0}};

        (* DONT_TOUCH = "TRUE" *)
        LUT6 #(
            .INIT(INIT_64)
        ) lut_inst (
            .O  (data_out_0),
            .I0 (padded_in[0]),
            .I1 (padded_in[1]),
            .I2 (padded_in[2]),
            .I3 (padded_in[3]),
            .I4 (padded_in[4]),
            .I5 (padded_in[5])
        );
    end
    else begin : gen_unsupported
        initial $error("structural_dwn_lut_neuron: LUT_N > 6 not supported");
    end
endgenerate

endmodule

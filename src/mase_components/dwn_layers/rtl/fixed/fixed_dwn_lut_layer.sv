/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
module fixed_dwn_lut_layer #(
    parameter INPUT_SIZE  = 8,
    parameter OUTPUT_SIZE = 4,
    parameter LUT_N       = 2,
    // INDEX_BITS: bits per index field = ceil(log2(INPUT_SIZE)).
    // Must be set to match _pack_dwn_lut_params in passes.py (= $clog2(INPUT_SIZE)).
    // Using an explicit parameter (rather than $clog2 inline) avoids tool issues with
    // dependent-parameter array dimensions in older Verilator/VCS versions.
    parameter INDEX_BITS  = $clog2(INPUT_SIZE > 1 ? INPUT_SIZE : 2),
    // INPUT_INDICES[(i*LUT_N + k)*INDEX_BITS +: INDEX_BITS] = index of input k for LUT i
    parameter [OUTPUT_SIZE*LUT_N*INDEX_BITS-1:0]    INPUT_INDICES = {(OUTPUT_SIZE*LUT_N*INDEX_BITS){1'b0}},
    // LUT_CONTENTS[i*(2**LUT_N) +: (2**LUT_N)] = contents for LUT i
    parameter [OUTPUT_SIZE*(2**LUT_N)-1:0] LUT_CONTENTS  = {(OUTPUT_SIZE*(2**LUT_N)){1'b0}}
) (
    input  wire clk,
    input  wire rst,

    input  wire  [INPUT_SIZE-1:0]  data_in_0,
    input  logic                   data_in_0_valid,
    output logic                   data_in_0_ready,

    output wire  [OUTPUT_SIZE-1:0] data_out_0,
    output logic                   data_out_0_valid,
    input  logic                   data_out_0_ready
);

    genvar i, k;
    generate
        for (i = 0; i < OUTPUT_SIZE; i = i + 1)
        begin : gen_lut
            wire [LUT_N-1:0] lut_inputs;

            for (k = 0; k < LUT_N; k = k + 1)
            begin : gen_inputs
                wire [INDEX_BITS-1:0] idx;
                assign idx           = INPUT_INDICES[(i*LUT_N + k)*INDEX_BITS +: INDEX_BITS];
                assign lut_inputs[k] = data_in_0[idx];
            end

            fixed_dwn_lut_neuron #(
                .LUT_N       (LUT_N),
                .LUT_CONTENTS(LUT_CONTENTS[i*(2**LUT_N) +: (2**LUT_N)])
            ) lut_inst (
                .data_in_0 (lut_inputs),
                .data_out_0(data_out_0[i])
            );
        end
    endgenerate

    assign data_out_0_valid = data_in_0_valid;
    assign data_in_0_ready  = data_out_0_ready;

endmodule

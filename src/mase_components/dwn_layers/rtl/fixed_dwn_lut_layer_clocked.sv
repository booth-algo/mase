/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
module fixed_dwn_lut_layer_clocked #(
    parameter INPUT_SIZE  = 8,
    parameter OUTPUT_SIZE = 4,
    parameter LUT_N       = 2,
    // INPUT_INDICES[(i*LUT_N + k)*8 +: 8] = index of input k for LUT i
    parameter [OUTPUT_SIZE*LUT_N*8-1:0]    INPUT_INDICES = {(OUTPUT_SIZE*LUT_N*8){1'b0}},
    // LUT_CONTENTS[i*(2**LUT_N) +: (2**LUT_N)] = contents for LUT i
    parameter [OUTPUT_SIZE*(2**LUT_N)-1:0] LUT_CONTENTS  = {(OUTPUT_SIZE*(2**LUT_N)){1'b0}}
) (
    input  wire clk,
    input  wire rst,

    input  wire  [INPUT_SIZE-1:0]  data_in_0,
    input  logic                   data_in_0_valid,
    output logic                   data_in_0_ready,

    output logic [OUTPUT_SIZE-1:0] data_out_0,
    output logic                   data_out_0_valid,
    input  logic                   data_out_0_ready
);

    // Combinational LUT outputs (before register stage)
    wire [OUTPUT_SIZE-1:0] comb_out;

    genvar i, k;
    generate
        for (i = 0; i < OUTPUT_SIZE; i = i + 1)
        begin : gen_lut
            wire [LUT_N-1:0] lut_inputs;

            for (k = 0; k < LUT_N; k = k + 1)
            begin : gen_inputs
                wire [7:0] idx;
                assign idx           = INPUT_INDICES[(i*LUT_N + k)*8 +: 8];
                assign lut_inputs[k] = data_in_0[idx];
            end

            fixed_dwn_lut_neuron #(
                .LUT_N       (LUT_N),
                .LUT_CONTENTS(LUT_CONTENTS[i*(2**LUT_N) +: (2**LUT_N)])
            ) lut_inst (
                .data_in_0 (lut_inputs),
                .data_out_0(comb_out[i])
            );
        end
    endgenerate

    // Output register stage: register data and valid on posedge clk
    always_ff @(posedge clk) begin
        if (rst) begin
            data_out_0       <= {OUTPUT_SIZE{1'b0}};
            data_out_0_valid <= 1'b0;
        end else begin
            data_out_0       <= comb_out;
            data_out_0_valid <= data_in_0_valid;
        end
    end

    // Simple pass-through ready (no backpressure buffer)
    assign data_in_0_ready = data_out_0_ready;

endmodule

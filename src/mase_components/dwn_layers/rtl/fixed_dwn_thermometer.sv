module fixed_dwn_thermometer #(
    parameter NUM_FEATURES   = 4,   // F - number of features
    parameter FEATURE_WIDTH  = 8,   // W - bits per feature
    parameter NUM_THRESHOLDS = 8,   // T - thresholds per feature

    // THRESHOLDS[(f*NUM_THRESHOLDS + t)*FEATURE_WIDTH +: FEATURE_WIDTH] = threshold for feature f, level t
    parameter [(NUM_FEATURES*NUM_THRESHOLDS*FEATURE_WIDTH)-1:0] THRESHOLDS = '0
) (
    // Packed input: feature f at data_in_0[f*FEATURE_WIDTH +: FEATURE_WIDTH]
    input  wire [NUM_FEATURES*FEATURE_WIDTH-1:0] data_in_0,
    input  logic data_in_0_valid,
    output logic data_in_0_ready,

    // Packed output: bit [f*NUM_THRESHOLDS + t] = (feature_f >= threshold_f_t)
    output logic [NUM_FEATURES*NUM_THRESHOLDS-1:0] data_out_0,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

    genvar f, t;
    generate
        for (f = 0; f < NUM_FEATURES; f = f + 1)
        begin : gen_feature
            wire [FEATURE_WIDTH-1:0] feature_val;
            assign feature_val = data_in_0[f*FEATURE_WIDTH +: FEATURE_WIDTH];

            for (t = 0; t < NUM_THRESHOLDS; t = t + 1)
            begin : gen_threshold
                wire [FEATURE_WIDTH-1:0] threshold_val;
                assign threshold_val = THRESHOLDS[(f*NUM_THRESHOLDS + t)*FEATURE_WIDTH +: FEATURE_WIDTH];
                assign data_out_0[f*NUM_THRESHOLDS + t] = (feature_val >= threshold_val) ? 1'b1 : 1'b0;
            end
        end
    endgenerate

    assign data_out_0_valid = data_in_0_valid;
    assign data_in_0_ready  = data_out_0_ready;

endmodule

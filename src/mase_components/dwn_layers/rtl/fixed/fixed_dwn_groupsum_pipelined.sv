// Pipelined GroupSum: 2-stage popcount replaces combinational $countones.
// Stage 1: split each group into ~10-bit sub-groups, popcount each, register.
// Stage 2: sum partial counts per group, register final output.
`timescale 1ns/1ps
module fixed_dwn_groupsum_pipelined #(
    parameter INPUT_SIZE     = 4,
    parameter NUM_GROUPS     = 2,
    parameter SUB_GROUP_SIZE = 10
) (
    input  wire                                    clk,
    input  wire                                    rst,

    input  wire  [INPUT_SIZE-1:0]                  data_in_0,
    input  logic                                   data_in_0_valid,
    output logic                                   data_in_0_ready,

    output logic [$clog2(INPUT_SIZE/NUM_GROUPS):0] data_out_0 [0:NUM_GROUPS-1],
    output logic                                   data_out_0_valid,
    input  logic                                   data_out_0_ready
);

    localparam GROUP_SIZE      = INPUT_SIZE / NUM_GROUPS;
    // Number of full sub-groups per group
    localparam NUM_SUB_FULL    = GROUP_SIZE / SUB_GROUP_SIZE;
    // Remainder bits after full sub-groups
    localparam REMAINDER       = GROUP_SIZE - NUM_SUB_FULL * SUB_GROUP_SIZE;
    // Total sub-groups per group (add 1 if remainder exists)
    localparam NUM_SUB_GROUPS  = (REMAINDER > 0) ? NUM_SUB_FULL + 1 : NUM_SUB_FULL;
    // Bit width for a sub-group popcount (max value = SUB_GROUP_SIZE)
    localparam SUB_COUNT_WIDTH = $clog2(SUB_GROUP_SIZE) + 1;
    // Bit width for the final group count
    localparam COUNT_WIDTH     = $clog2(GROUP_SIZE) + 1;

    // ----------------------------------------------------------------
    // Stage 1: sub-group popcounts (combinational) -> registered
    // ----------------------------------------------------------------
    // Partial counts: [group][sub_group]
    logic [SUB_COUNT_WIDTH-1:0] partial_comb [0:NUM_GROUPS-1][0:NUM_SUB_GROUPS-1];
    logic [SUB_COUNT_WIDTH-1:0] partial_reg  [0:NUM_GROUPS-1][0:NUM_SUB_GROUPS-1];
    logic                       valid_s1;

    genvar g, s;
    generate
        for (g = 0; g < NUM_GROUPS; g = g + 1) begin : gen_group
            // Full-sized sub-groups
            for (s = 0; s < NUM_SUB_FULL; s = s + 1) begin : gen_sub
                wire [SUB_GROUP_SIZE-1:0] sub_data;
                assign sub_data = data_in_0[g*GROUP_SIZE + s*SUB_GROUP_SIZE +: SUB_GROUP_SIZE];
                always_comb begin
                    partial_comb[g][s] = $countones(sub_data);
                end
            end
            // Remainder sub-group (if GROUP_SIZE not evenly divisible)
            if (REMAINDER > 0) begin : gen_remainder
                wire [REMAINDER-1:0] rem_data;
                assign rem_data = data_in_0[g*GROUP_SIZE + NUM_SUB_FULL*SUB_GROUP_SIZE +: REMAINDER];
                always_comb begin
                    partial_comb[g][NUM_SUB_FULL] = $countones(rem_data);
                end
            end
        end
    endgenerate

    // Register Stage 1 outputs
    always_ff @(posedge clk) begin
        if (rst) begin
            valid_s1 <= 1'b0;
            for (int gi = 0; gi < NUM_GROUPS; gi = gi + 1)
                for (int si = 0; si < NUM_SUB_GROUPS; si = si + 1)
                    partial_reg[gi][si] <= '0;
        end else begin
            valid_s1 <= data_in_0_valid;
            for (int gi = 0; gi < NUM_GROUPS; gi = gi + 1)
                for (int si = 0; si < NUM_SUB_GROUPS; si = si + 1)
                    partial_reg[gi][si] <= partial_comb[gi][si];
        end
    end

    // ----------------------------------------------------------------
    // Stage 2: sum partial counts per group (combinational) -> registered
    // ----------------------------------------------------------------
    logic [COUNT_WIDTH-1:0] sum_comb [0:NUM_GROUPS-1];

    always_comb begin
        for (int gi = 0; gi < NUM_GROUPS; gi = gi + 1) begin
            sum_comb[gi] = '0;
            for (int si = 0; si < NUM_SUB_GROUPS; si = si + 1)
                sum_comb[gi] = sum_comb[gi] + COUNT_WIDTH'(partial_reg[gi][si]);
        end
    end

    // Register Stage 2 outputs
    always_ff @(posedge clk) begin
        if (rst) begin
            data_out_0_valid <= 1'b0;
            for (int gi = 0; gi < NUM_GROUPS; gi = gi + 1)
                data_out_0[gi] <= '0;
        end else begin
            data_out_0_valid <= valid_s1;
            for (int gi = 0; gi < NUM_GROUPS; gi = gi + 1)
                data_out_0[gi] <= sum_comb[gi][$clog2(INPUT_SIZE/NUM_GROUPS):0];
        end
    end

    // Simple pass-through ready (no backpressure buffer)
    assign data_in_0_ready = data_out_0_ready;

endmodule

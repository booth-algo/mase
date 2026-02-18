module fixed_dwn_groupsum #(
    parameter INPUT_SIZE = 4,
    parameter NUM_GROUPS = 2
) (
    input  wire  [INPUT_SIZE-1:0]                  data_in_0,
    input  logic                                   data_in_0_valid,
    output logic                                   data_in_0_ready,

    output logic [$clog2(INPUT_SIZE/NUM_GROUPS):0] data_out_0 [0:NUM_GROUPS-1],
    output logic                                   data_out_0_valid,
    input  logic                                   data_out_0_ready
);

    localparam GROUP_SIZE = INPUT_SIZE / NUM_GROUPS;

    genvar i;
    generate
        for (i = 0; i < NUM_GROUPS; i = i + 1)
        begin : gen_group
            wire [GROUP_SIZE-1:0] group_data;
            assign group_data = data_in_0[i*GROUP_SIZE +: GROUP_SIZE];

            always_comb
            begin
                data_out_0[i] = $countones(group_data);
            end
        end
    endgenerate

    assign data_out_0_valid = data_in_0_valid;
    assign data_in_0_ready  = data_out_0_ready;

endmodule

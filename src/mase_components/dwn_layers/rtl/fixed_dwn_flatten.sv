module fixed_dwn_flatten #(
    parameter IN_COLS = 2,
    parameter IN_ROWS = 2
) (
    input  wire [IN_COLS-1:0]         data_in_0 [0:IN_ROWS-1],
    input  wire                       data_in_0_valid,
    output wire                       data_in_0_ready,

    output wire [IN_COLS*IN_ROWS-1:0] data_out_0,
    output wire                       data_out_0_valid,
    input  wire                       data_out_0_ready
);

    genvar i, j;
    generate
        for (i = 0; i < IN_ROWS; i = i + 1)
        begin : gen_row
            for (j = 0; j < IN_COLS; j = j + 1)
            begin : gen_col
                assign data_out_0[i*IN_COLS + j] = data_in_0[i][j];
            end
        end
    endgenerate

    assign data_out_0_valid = data_in_0_valid;
    assign data_in_0_ready  = data_out_0_ready;

endmodule

`timescale 1ns/1ps
module tb;
    localparam N = 8;
    localparam WIDTH = 32;
    localparam STEPS = 3000;
    reg clk = 0;
    reg rst = 1;
    reg signed [WIDTH-1:0] drive_mem [0:STEPS-1];
    reg signed [WIDTH-1:0] drive;
    wire [N-1:0] detected;

    spikeling_resonators dut (.clk(clk), .rst(rst), .drive(drive), .detected(detected));
    always #5 clk = ~clk;

    integer t;
    initial begin
        $readmemh("fib_drive.hex", drive_mem);
        @(negedge clk); rst = 0;
        for (t = 0; t < STEPS; t = t + 1) begin
            drive = drive_mem[t];
            @(negedge clk);
        end
        for (t = 0; t < N; t = t + 1) begin
            $display("XFINAL %0d %0d", t, $signed(dut.x[t]));
        end
        $finish;
    end
endmodule

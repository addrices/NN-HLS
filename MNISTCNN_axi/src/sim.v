`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/19/2022 10:58:37 AM
// Design Name: 
// Module Name: sim
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module sim();
reg clk;
reg rstn;
reg [4:0]axi_araddr;
wire axi_arready;
reg axi_arvalid;
reg [4:0]axi_awaddr;
wire axi_awready;
reg axi_awvalid;
reg axi_bready;
wire [1:0]axi_bresp;
wire axi_bvalid;
wire [31:0]axi_rdata;
reg axi_rready;
wire [1:0]axi_rresp;
wire axi_rvalid;
reg [31:0]axi_wdata;
wire axi_wready;
reg [3:0]axi_wstrb;
reg axi_wvalid;
reg clk;
reg rstn;

reg [1:0] state_r;
always #10 clk = ~clk;

reg [7:0] i;
test_design_wrapper testd(
  .axi_araddr(axi_araddr),
  .axi_arready(axi_arready),
  .axi_arvalid(axi_arvalid),
  .axi_awaddr(axi_awaddr),
  .axi_awready(axi_awready),
  .axi_awvalid(axi_awvalid),
  .axi_bready(axi_bready),
  .axi_bresp(axi_bresp),
  .axi_bvalid(axi_bvalid),
  .axi_rdata(axi_rdata),
  .axi_rready(axi_rready),
  .axi_rresp(axi_rresp),
  .axi_rvalid(axi_rvalid),
  .axi_wdata(axi_wdata),
  .axi_wready(axi_wready),
  .axi_wstrb(axi_wstrb),
  .axi_wvalid(axi_wvalid),
  .clk(clk),
  .rstn(rstn)
);
initial begin
  #1 clk = 0;
  #2 rstn = 0;
     axi_araddr = 0;
     axi_arvalid = 0;
     axi_awaddr = 0;
     axi_awvalid = 0;
     axi_bready = 0;
     axi_rready = 0;
     axi_wdata = 0;
     axi_wstrb = 4'hf;
  #30 rstn = 1;
  #10 axi_awvalid = 1;
      axi_awaddr = 32'h40020018;
  for(i = 0;axi_awvalid!=1 | axi_arready!=1;i = i+1)begin
    #20;
  end
    #20 axi_awvalid = 0;
        axi_wvalid  = 1;
        axi_wdata   = 32'h40028100;
  for(i = 0;axi_wvalid!=1 | axi_wready!=1;i = i+1)begin
    #20;
  end
    #20 axi_wvalid = 0;
        axi_bready = 1;
  for(i = 0;axi_bready!=1 | axi_bvalid!=1;i = i+1)begin
    #20;
  end
    #20 axi_awvalid = 1;
      axi_awaddr = 32'h40020000;
  for(i = 0;axi_awvalid!=1 | axi_arready!=1;i = i+1)begin
    #20;
  end
    #20 axi_awvalid = 0;
        axi_wvalid  = 1;
        axi_wdata   = 32'h1;
  for(i = 0;axi_wvalid!=1 | axi_wready!=1;i = i+1)begin
    #20;
  end
    #20 axi_wvalid = 0;
        axi_bready = 1;
  for(i = 0;axi_bready!=1 | axi_bvalid!=1;i = i+1)begin
    #20;
  end
    #20;
    
 end

endmodule

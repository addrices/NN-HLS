############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2021 Xilinx, Inc. All Rights Reserved.
############################################################

open_project HLS_proj
set_top top
add_files ../src/top.cpp
add_files -tb ../src/test.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xc7a100tcsg324-1}
create_clock -period 10 -name default
config_interface -m_axi_addr64=0
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
exit

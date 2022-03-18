############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2021 Xilinx, Inc. All Rights Reserved.
############################################################
open_project HLS_proj
open_solution "solution1" -flow_target vivado
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
exit
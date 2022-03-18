############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
set proj_name HLS_proj
set hls_src ../src

open_project -reset $proj_name
set_top top
add_files $hls_src/top.cpp
add_files -tb $hls_src/test.cpp

open_solution -reset "solution1"
# set_part {xc7z020clg400-1} -tool vivado
create_clock -period 10 -name default

exit

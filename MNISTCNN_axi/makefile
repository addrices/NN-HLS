.PHONY: clean

LOCAL := $(shell pwd)
NN_H_DIR := $(LOCAL)/../nn-h
SRC_DIR := $(LOCAL)/src
OUT_DIR := $(LOCAL)/output
NN_H_FILE := $(shell find $(NN_H_DIR) -name "*.h")
SRC_FILE := $(SRC_DIR)/top.cpp
SRC_TEST_FILE := $(SRC_DIR)/test.cpp

HLS_PROJ_SCRIPT := $(LOCAL)/script/create_hlspro.tcl
HLS_IP_SCRIPT := $(LOCAL)/script/make_ip.tcl
VIVADO_PROJ_SCRIPT := $(LOCAL)/script/create_vivado.tcl
HLS_PROJ_NAME := HLS_proj

BD_DIR := $(OUT_DIR)/bd
BD_FILE := $(SRC_DIR)/design_1.bd


hls_proj:$(NN_H_FILE) $(SRC_FILE) $(HLS_PROJ_SCRIPT)
	@mkdir -p $(OUT_DIR)
	@cd $(OUT_DIR) && vitis_hls -f $(HLS_PROJ_SCRIPT)

hls_ip: hls_proj 
	@cd $(OUT_DIR) && vitis_hls -f $(HLS_IP_SCRIPT)


viv_pro:hls_ip
	@mkdir -p $(BD_DIR)
	@cp $(BD_FILE) $(BD_DIR)
	@cd $(OUT_DIR) && vivado -mode batch -source $(VIVADO_PROJ_SCRIPT)

clean:
	rm $(OUT_DIR) -r
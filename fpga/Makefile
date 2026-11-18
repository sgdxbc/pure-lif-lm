# Build the FPGA kernel and its host program with Vitis (on a HACC build machine; docs/experiments.md 5.44).
#   make PACK=../runs/g29_hw/fpga NB=4 TARGET=hw_emu      # emulation of the hardware (RTL simulation; slow, small runs)
#   make PACK=../runs/g29_hw/fpga NB=16 TARGET=hw         # the bitstream for the board (hours)
#   make host PACK=... NB=...
# PACK: fpga/pack.py's output (model_cfg.h fixes the network's sizes at compile time). NB: streams. The platform is the
# U55C shell installed at HACC@NUS (github.com/Xtra-Computing/hacc_demo); override PLATFORM for another one.
PLATFORM ?= xilinx_u55c_gen3x16_xdma_3_202210_1
PACK ?= ../runs/g29_hw/fpga
NB ?= 16
TARGET ?= hw
BUILD := build/$(TARGET)_nb$(NB)

all: $(BUILD)/lif_lm.xclbin host

$(BUILD)/lif_lm.xo: lif_lm.cpp $(PACK)/model_cfg.h
	mkdir -p $(BUILD)
	v++ -c -t $(TARGET) --platform $(PLATFORM) -k lif_lm -I$(PACK) -DNB=$(NB) --temp_dir $(BUILD)/tmp -o $@ lif_lm.cpp

$(BUILD)/lif_lm.xclbin: $(BUILD)/lif_lm.xo connectivity.cfg
	v++ -l -t $(TARGET) --platform $(PLATFORM) --config connectivity.cfg --temp_dir $(BUILD)/tmp -o $@ $<

host: host.cpp $(PACK)/model_cfg.h
	g++ -std=c++17 -O2 -DNB=$(NB) -I$(XILINX_XRT)/include -I$(PACK) -o host_nb$(NB) host.cpp \
	    -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread

clean:
	rm -rf build host_nb* *.log _x .Xil

.PHONY: all host clean

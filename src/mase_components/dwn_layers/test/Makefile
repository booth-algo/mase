# Makefile for running the cocotb testbenches on dwn_layers

TOPLEVEL_LANG = verilog

VERILOG_SOURCES = $(shell pwd)/../../rtl/*.sv

TOPLEVEL ?= fixed_dwn_groupsum
MODULE   ?= fixed_dwn_groupsum_tb

SIM ?= verilator

include $(shell cocotb-config --makefiles)/Makefile.sim

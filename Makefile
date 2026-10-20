# =============================================================================
# Makefile — Kalman Filter Milestone 4
# Builds RVV vectorised LKF/EKF executables and provides run/CSV/verify helpers.
#
# Expected main assembly files:
#   lkf_vector.s
#   ekf_vector.s
#
# The C/C++ driver files are auto-detected. If detection picks the wrong file,
# override them from the command line, for example:
#   make LKF_DRIVER=src/lkf_vector_main.c EKF_DRIVER=src/ekf_vector_main.c
#   make LKF_DRIVER=lkf_main.c EKF_DRIVER=ekf_main.c
#
# If EKF scalar helper functions are split into a separate assembly file, pass it:
#   make EKF_EXTRA_SRC=ekf_helpers.s
#
# If you run on an emulator, use RUNNER, for example:
#   make run RUNNER="qemu-riscv64 -L /usr/riscv64-linux-gnu"
# =============================================================================

# ----------------------------- Toolchain -------------------------------------
CROSS_COMPILE ?= riscv64-unknown-linux-gnu-
CC            ?= $(CROSS_COMPILE)gcc
CXX           ?= $(CROSS_COMPILE)g++
OBJDUMP       ?= $(CROSS_COMPILE)objdump
PYTHON        ?= python3
RUNNER        ?=

# RVV + double precision floating point ABI.
ARCH ?= rv64gcv
ABI  ?= lp64d
OPT  ?= -O2

COMMON_FLAGS := $(OPT) -march=$(ARCH) -mabi=$(ABI) -fno-fast-math -Wall -Wextra
CFLAGS       += $(COMMON_FLAGS) -std=c11
CXXFLAGS     += $(COMMON_FLAGS) -std=c++17
ASFLAGS      += $(COMMON_FLAGS)
LDLIBS       += -lm

# Set STATIC=1 if your emulator/system needs fully static RISC-V binaries.
STATIC ?= 0
ifeq ($(STATIC),1)
LDFLAGS += -static
endif

# ----------------------------- Files -----------------------------------------
BUILD_DIR ?= build
OUT_DIR   ?= output
VERIFY_DIR ?= m4_verification_tables

LKF_VECTOR_ASM ?= lkf_vector.s
EKF_VECTOR_ASM ?= ekf_vector.s

# Auto-detect drivers. Override from command line if needed.
LKF_DRIVER ?= $(firstword $(wildcard \
  lkf_vector_main.c lkf_vector_main.cpp \
  lkf_main.c lkf_main.cpp \
  src/lkf_vector_main.c src/lkf_vector_main.cpp \
  src/lkf_main.c src/lkf_main.cpp \
  src/lkf.c src/lkf.cpp))

EKF_DRIVER ?= $(firstword $(wildcard \
  ekf_vector_main.c ekf_vector_main.cpp \
  ekf_main.c ekf_main.cpp \
  src/ekf_vector_main.c src/ekf_vector_main.cpp \
  src/ekf_main.c src/ekf_main.cpp \
  src/ekf.c src/ekf.cpp))

# Extra source files, for scalar helper functions only.
# IMPORTANT: do not include files that also define lkf_predict/lkf_update or
# ekf_predict/ekf_update, otherwise the linker will report duplicate symbols.
LKF_EXTRA_SRC ?=
EKF_EXTRA_SRC ?=

LKF_EXE ?= lkf_vector
EKF_EXE ?= ekf_vector

# Required CSVs for verification.
M3_LKF_CSV ?= $(OUT_DIR)/lkf_asm_output.csv
M3_EKF_CSV ?= $(OUT_DIR)/ekf_asm_output.csv
M4_LKF_CSV ?= $(OUT_DIR)/lkf_vector_output.csv
M4_EKF_CSV ?= $(OUT_DIR)/ekf_vector_output.csv

# Convert source paths to build object paths, e.g. src/a.cpp -> build/src/a.o.
obj = $(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(basename $(1))))

LKF_SRCS := $(LKF_DRIVER) $(LKF_VECTOR_ASM) $(LKF_EXTRA_SRC)
EKF_SRCS := $(EKF_DRIVER) $(EKF_VECTOR_ASM) $(EKF_EXTRA_SRC)
LKF_OBJS := $(call obj,$(LKF_SRCS))
EKF_OBJS := $(call obj,$(EKF_SRCS))

# ----------------------------- Targets ---------------------------------------
.PHONY: all clean run run-lkf run-ekf csv verify dump print-config help \
        check-lkf-driver check-ekf-driver check-vector-files

all: check-vector-files $(LKF_EXE) $(EKF_EXE)

$(LKF_EXE): check-lkf-driver $(LKF_OBJS)
	$(CXX) $(COMMON_FLAGS) $(LDFLAGS) $(LKF_OBJS) -o $@ $(LDLIBS)

$(EKF_EXE): check-ekf-driver $(EKF_OBJS)
	$(CXX) $(COMMON_FLAGS) $(LDFLAGS) $(EKF_OBJS) -o $@ $(LDLIBS)

# ----------------------------- Compile rules ---------------------------------
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.s
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -c $< -o $@

# ----------------------------- Checks ----------------------------------------
check-vector-files:
	@test -f "$(LKF_VECTOR_ASM)" || (echo "ERROR: missing $(LKF_VECTOR_ASM)"; exit 1)
	@test -f "$(EKF_VECTOR_ASM)" || (echo "ERROR: missing $(EKF_VECTOR_ASM)"; exit 1)

check-lkf-driver:
	@test -n "$(LKF_DRIVER)" || (echo "ERROR: no LKF driver found. Run: make LKF_DRIVER=path/to/lkf_driver.c"; exit 1)
	@test -f "$(LKF_DRIVER)" || (echo "ERROR: LKF_DRIVER='$(LKF_DRIVER)' does not exist"; exit 1)

check-ekf-driver:
	@test -n "$(EKF_DRIVER)" || (echo "ERROR: no EKF driver found. Run: make EKF_DRIVER=path/to/ekf_driver.c"; exit 1)
	@test -f "$(EKF_DRIVER)" || (echo "ERROR: EKF_DRIVER='$(EKF_DRIVER)' does not exist"; exit 1)

# ----------------------------- Run helpers -----------------------------------
run: run-lkf run-ekf

run-lkf: $(LKF_EXE)
	@mkdir -p $(OUT_DIR)
	$(RUNNER) ./$(LKF_EXE)

run-ekf: $(EKF_EXE)
	@mkdir -p $(OUT_DIR)
	$(RUNNER) ./$(EKF_EXE)

# Runs executables and standardises outputs as output/lkf_vector_output.csv and
# output/ekf_vector_output.csv using the helper script.
csv: $(LKF_EXE) $(EKF_EXE)
	@mkdir -p $(OUT_DIR)
	$(PYTHON) generate_m4_output_csvs.py \
	  --lkf-cmd "$(RUNNER) ./$(LKF_EXE)" \
	  --ekf-cmd "$(RUNNER) ./$(EKF_EXE)" \
	  --out-dir "$(OUT_DIR)"

# Compares M4 vector CSVs against M3 scalar CSVs.
verify:
	$(PYTHON) verify_m4.py \
	  --m3-lkf "$(M3_LKF_CSV)" \
	  --m4-lkf "$(M4_LKF_CSV)" \
	  --m3-ekf "$(M3_EKF_CSV)" \
	  --m4-ekf "$(M4_EKF_CSV)" \
	  --out-dir "$(VERIFY_DIR)" \
	  --tol 1e-9

# Static disassembly for instruction-count discussion in the report.
dump: $(LKF_EXE) $(EKF_EXE)
	@mkdir -p $(BUILD_DIR)/dump
	$(OBJDUMP) -d $(BUILD_DIR)/$(basename $(LKF_VECTOR_ASM)).o > $(BUILD_DIR)/dump/lkf_vector.dump
	$(OBJDUMP) -d $(BUILD_DIR)/$(basename $(EKF_VECTOR_ASM)).o > $(BUILD_DIR)/dump/ekf_vector.dump
	@echo "Wrote $(BUILD_DIR)/dump/lkf_vector.dump and $(BUILD_DIR)/dump/ekf_vector.dump"

print-config:
	@echo "CROSS_COMPILE = $(CROSS_COMPILE)"
	@echo "CC            = $(CC)"
	@echo "CXX           = $(CXX)"
	@echo "ARCH          = $(ARCH)"
	@echo "ABI           = $(ABI)"
	@echo "LKF_DRIVER    = $(LKF_DRIVER)"
	@echo "EKF_DRIVER    = $(EKF_DRIVER)"
	@echo "LKF_EXTRA_SRC = $(LKF_EXTRA_SRC)"
	@echo "EKF_EXTRA_SRC = $(EKF_EXTRA_SRC)"
	@echo "RUNNER        = $(RUNNER)"

clean:
	rm -rf $(BUILD_DIR) $(LKF_EXE) $(EKF_EXE)

help:
	@echo "Milestone 4 Makefile"
	@echo ""
	@echo "Common commands:"
	@echo "  make print-config"
	@echo "  make all"
	@echo "  make run"
	@echo "  make csv"
	@echo "  make verify"
	@echo "  make dump"
	@echo "  make clean"
	@echo ""
	@echo "Useful overrides:"
	@echo "  make LKF_DRIVER=lkf_main.c EKF_DRIVER=ekf_main.c"
	@echo "  make EKF_EXTRA_SRC=ekf_helpers.s"
	@echo "  make RUNNER='qemu-riscv64 -L /usr/riscv64-linux-gnu' run"
	@echo "  make STATIC=1 all"

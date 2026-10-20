# RISC-V Kalman Filters for 3D Human Motion

An implementation of Linear and Extended Kalman Filters for a 23-joint motion-capture model. The full state contains 276 values (12 states per joint), and the project includes reusable RISC-V scalar kernels, RVV vector kernels, C drivers, verification tooling, diagnostic plots, and a skeletal animation.

## Highlights

- 276-dimensional state across 23 joints
- Linear Kalman Filter (LKF) and Extended Kalman Filter (EKF)
- RV64D scalar assembly with reusable matrix primitives
- RISC-V Vector Extension (RVV) implementations
- LP64D ABI-compatible assembly routines
- Cholesky-based solving and Kalman gain computation
- Python scripts for output normalization and numerical verification

## Repository layout

```text
.
├── ekf_asm.s                 # Scalar EKF and matrix kernels
├── lkf_asm.s                 # Scalar LKF and matrix kernels
├── ekf_vector.s              # RVV EKF implementation
├── lkf_vector.s              # RVV LKF implementation
├── ekf_main.c                # EKF driver
├── lkf_main.c                # LKF driver
├── generate_m4_output_csvs.py
├── verify_m4.py
└── Makefile
```

## Requirements

- A RISC-V GNU toolchain with RVV support
- An RV64GCV-capable target or emulator such as QEMU
- Python 3 with NumPy and pandas for verification

The C drivers expect a motion-capture input file at `data/noisy.csv`. The dataset is not included in this repository.

## Build

```bash
make print-config
make
```

To run through QEMU:

```bash
make RUNNER="qemu-riscv64 -L /usr/riscv64-linux-gnu" run
```

To generate standardized CSV outputs and compare vector results with scalar baselines:

```bash
make csv
make verify
```

Toolchain, driver, runner, and static-linking options can be overridden through the variables documented by `make help`.

## Results

The verification scripts generate per-joint and per-state error tables for comparing scalar and vector outputs. Diagnostic plots and animations can be produced locally from the generated CSV files.

## Notes

Generated binaries, output CSVs, local datasets, and build products are intentionally excluded from version control.

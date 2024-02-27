# FPGA-accelerated JSON parsing

This project was created as part of the Data Processing on Modern Hardware seminar 2023 / 2024.

## Clone the Project and Setup the Build Directory

1. Clone the repository: `git clone --recursive git@github.com:phkeese/fpga-json-parser.git`
1. Enter the Hyrise directory: `cd fpga-json-parser`
1. Create the build directory: `mkdir cmake-build-debug && cd cmake-build-debug`
1. Generate Makefiles: `cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10_usm`

## Build and run the Parser Emulator
1. Build the Parser: `make emu`
1. Run the Parser: `./json_parser.emu [JSON_FILE]`

## Build and run the Benchmarks Emulator
1. Build the Parser: `make bench_emu`
1. Run the Parser: `./json_parser.bench_emu`

## Build and run the Parser on FPGA Hardware
1. Build the Parser: `make fpga`
1. Run the Parser: `./json_parser.fpga [JSON_FILE]`

## Build and run the Benchmarks on FPGA Hardware
1. Build the Parser: `make bench_fpga`
1. Run the Parser: `./json_parser.bench_fpga`

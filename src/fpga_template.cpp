#include <iomanip>
#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include <array>
#include <chrono>
#include <random>
#include <vector>

constexpr auto ROWS = size_t{256};
// One column takes ROWS steps, but after each element, the next column can
// start. So, we allow ROWS pipeline steps to start another column after each
// element.
constexpr auto PIPELINE_DEPTH = ROWS;

using Element = uint8_t;
using Column = std::array<Element, ROWS>;
using Matrix = std::vector<Column>;
using Duration = std::chrono::nanoseconds;

Matrix generate_input(size_t column_count) {
  std::random_device rd;
  std::uniform_int_distribution<Element> dist(0, 255);

  auto data = Matrix(column_count);
  for (auto &column : data) {
    for (auto &element : column) {
      element = dist(rd);
    }
  }
  return data;
}

/**
 * Perform a highly data-dependent operation on M.
 * Reference implementation for validation.
 * @param O Matrix to operate on.
 */
void matrix_sum(const Matrix &I, Matrix &O) {
  assert(I.size() == O.size());
  // Cache last column.
  auto prev_in_row = Column{};
  const auto col_count = O.size();
  for (auto col_i = size_t{0}; col_i < col_count; col_i++) {
    // Cache previous element in column.
    auto prev_in_col = Element{};
    for (auto row_i = size_t{0}; row_i < ROWS; row_i++) {
      auto input = I[col_i][row_i];
      auto output = input + prev_in_row[row_i] + prev_in_col;
      O[col_i][row_i] = prev_in_row[row_i] = prev_in_col = output;
    }
  }
}

/**
 * Compute reference value on CPU.
 * @param input Data to compute result to.
 * @param output Output to write result into.
 */
Duration compute_cpu(const Matrix &input, Matrix &output) {
  const auto before = std::chrono::high_resolution_clock ::now();
  matrix_sum(input, output);
  const auto after = std::chrono::high_resolution_clock ::now();
  auto duration = after - before;
  return std::chrono::duration_cast<Duration>(duration);
}

/**
 * Compute on FPGA but sequentially, without pipelining.
 * @param q Queue to use.
 * @param input Data to compute result to.
 * @param output Output to write result into.
 * @returns Duration of computation.
 */
Duration compute_seq(sycl::queue &q, const Matrix &input, Matrix &output);

/**
 * Compute on FPGA but pipelined, hopefully faster.
 * @param q Queue to use.
 * @param input Data to compute result to.
 * @param output Output to write result into.
 * @returns Duration of computation.
 */
Duration compute_pipe(sycl::queue &q, const Matrix &input, Matrix &output);

/**
 * Validate two results are equal.
 * @param expect Expected value.
 * @param value Computed value.
 * @return Whether both are equal.
 */
bool validate(const Matrix &expect, const Matrix &value) {
  if (expect.size() != value.size()) {
    return false;
  }
  for (auto col_i = size_t{0}; col_i < expect.size(); col_i++) {
    for (auto row_i = size_t{0}; row_i < ROWS; row_i++) {
      if (expect[col_i][row_i] != value[col_i][row_i]) {
        return false;
      }
    }
  }
  return true;
}

void print(const Matrix &M);

int main(int argc, char **argv) {
  constexpr auto COLUMN_COUNT = ROWS * 10;
  auto column_count = COLUMN_COUNT;
  if (argc > 1) {
    errno = 0;
    auto count_arg = argv[1];
    column_count = std::strtoull(count_arg, nullptr, 10);
    if (errno) {
      perror("Parsing column count failed with error");
      std::cout << "Failed to parse column count, using fallback value.\n";
      column_count = COLUMN_COUNT;
    }
  }
  std::cout << "Executing with " << column_count << " columns.\n";

  bool passed = false;

  try {

    // Use compile-time macros to select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    //  - the simulator device
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Create inputs and outputs.
    const auto input = generate_input(column_count);
    //    print(input);
    auto output_cpu = Matrix(column_count);
    auto output_seq = Matrix(column_count);
    auto output_pipe = Matrix(column_count);

    // Run benchmarks.
    const auto duration_cpu = compute_cpu(input, output_cpu);
    const auto duration_seq = compute_seq(q, input, output_seq);
    const auto duration_pipe = compute_pipe(q, input, output_pipe);

    // Validate results.
    passed = true;
    if (!validate(output_cpu, output_seq)) {
      passed = false;
      std::cout << "Sequential: FAIL\n";
    } else {
      std::cout << "Sequential: PASS\n";
    }

    if (!validate(output_cpu, output_pipe)) {
      passed = false;
      std::cout << "Pipelined:  FAIL\n";
    } else {
      std::cout << "Pipelined:  PASS\n";
    }

    // Print out computation times.
    std::cout << "CPU took  " << duration_cpu.count() << " nanoseconds\n";
    std::cout << "Seq took  " << duration_seq.count() << " nanoseconds\n";
    std::cout << "Pipe took " << duration_pipe.count() << " nanoseconds\n ";

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code.
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

Duration compute_seq(sycl::queue &q, const Matrix &input, Matrix &output) {
  assert(input.size() == output.size());

  auto input_buffer = sycl::buffer(input);
  auto output_buffer = sycl::buffer(output);

  const auto col_count = output.size();

  auto before = std::chrono::high_resolution_clock ::now();
  q.submit([&](auto &h) {
    auto input_accessor = sycl::accessor(input_buffer, h, sycl::read_only);
    auto output_accessor = sycl::accessor(output_buffer, h, sycl::read_write);

    h.single_task([=]() {
      auto prev_in_row = Column{};
      for (auto col_i = size_t{0}; col_i < col_count; col_i++) {
        auto prev_in_col = Element{};
        for (auto row_i = size_t{0}; row_i < ROWS; row_i++) {
          auto input = input_accessor[col_i][row_i];
          auto output = input + prev_in_row[row_i] + prev_in_col;
          output_accessor[col_i][row_i] = prev_in_row[row_i] = prev_in_col =
              output;
        }
      }
    });
  });
  q.wait();
  auto after = std::chrono::high_resolution_clock ::now();
  return after - before;
}

template <size_t I> class PipeID;
template <size_t I>
using Pipe = sycl::ext::intel::pipe<PipeID<I>, Element, PIPELINE_DEPTH>;

// Feed all pipes with an element.
template <size_t RowId> void feed_pipe(Element value) {
  Pipe<RowId>::write(value);
  if constexpr (RowId < ROWS) {
    feed_pipe<RowId + 1>(value);
  }
}

template <size_t RowId, typename InAcc, typename OutAcc>
void compute_elements(Element &prev_in_col, const size_t col_id, InAcc &in_acc,
                      OutAcc &out_acc) {
  if constexpr (RowId < ROWS) {
    auto input = in_acc[col_id][RowId];
    auto prev_in_row = Pipe<RowId>::read();
    auto output = input + prev_in_row + prev_in_col;
    Pipe<RowId>::write(output);
    out_acc[col_id][RowId] = prev_in_col = output;
    compute_elements<RowId + 1>(prev_in_col, col_id, in_acc, out_acc);
  }
}

Duration compute_pipe(sycl::queue &q, const Matrix &input, Matrix &output) {
  assert(input.size() == output.size());

  auto input_buffer = sycl::buffer(input);
  auto output_buffer = sycl::buffer(output);

  const auto col_count = output.size();

  // Feed pipelines.
  q.single_task([=]() { feed_pipe<0>(Element{}); });
  auto before = std::chrono::high_resolution_clock ::now();
  q.submit([&](auto &h) {
    auto in_acc = sycl::accessor(input_buffer, h, sycl::read_only);
    auto out_acc = sycl::accessor(output_buffer, h, sycl::write_only);
    h.parallel_for(col_count, [=](auto col_id) {
      auto prev_in_col = Element{};
      compute_elements<0>(prev_in_col, col_id.get_linear_id(), in_acc, out_acc);
    });
  });
  q.wait();
  auto after = std::chrono::high_resolution_clock ::now();
  return after - before;
}

void print(const Matrix &M) {
  for (auto row_id = size_t{0}; row_id < ROWS; row_id++) {
    std::cout << "Row #" << row_id << ": ";
    for (auto col_id = size_t{0}; col_id < M.size(); col_id++) {
      std::cout << std::setfill('0') << std::setw(2) << std::right << std::hex
                << int(M[col_id][row_id]) << " ";
    }
    std::cout << "\n";
  }
}

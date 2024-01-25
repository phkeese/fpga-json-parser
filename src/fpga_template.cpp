#include <bitset>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

// oneAPI headers
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

std::vector<std::string_view> find_strings(sycl::queue &q,
                                           const std::string &input);

constexpr auto CACHE_LINE_SIZE = size_t{8};
constexpr auto PIPELINE_DEPTH = size_t{1};
using CacheLine = std::array<char, CACHE_LINE_SIZE>;
struct Bitmaps;

void start_kernels(sycl::queue &q, size_t count);
Bitmaps build_bitmaps(const CacheLine &cache_line);
int main(int argc, char **argv) {
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

    constexpr auto input = R"("k":"value", "k\"y": "\"", "key": "unescaped\"")";

    auto strings = find_strings(q, input);

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

  return EXIT_SUCCESS;
}

using InputPipe =
    sycl::ext::intel::experimental::pipe<class InputPipeID, CacheLine,
                                         PIPELINE_DEPTH>;

std::vector<std::string_view> find_strings(sycl::queue &q,
                                           const std::string &input) {
  auto cache_line_count = input.size() / CACHE_LINE_SIZE;
  start_kernels(q, cache_line_count + 1);

  // Send data to device.
  auto index = size_t{0};
  for (; index < cache_line_count; ++index) {
    auto begin = input.begin() + index * CACHE_LINE_SIZE;
    auto end = input.begin() + (index + 1) * CACHE_LINE_SIZE;
    auto line = CacheLine{};
    std::copy(begin, end, line.begin());
    InputPipe::write(q, line);
  }
  // Handle any overflowing bytes.
  auto begin = input.begin() + index * CACHE_LINE_SIZE;
  auto end = input.end();
  auto line = CacheLine{'\0'};
  std::copy(begin, end, line.begin());
  InputPipe::write(q, line);

  return std::vector<std::string_view>();
}

using Bitmap = std::bitset<CACHE_LINE_SIZE>;

struct OverflowBits {
  bool is_string;
  bool is_odd;
};

struct Bitmaps {
  Bitmap is_string;
  OverflowBits overflows;
};

const sycl::stream &operator<<(const sycl::stream &stream, const Bitmap &map) {
  for (auto index = size_t{0}; index < CACHE_LINE_SIZE; ++index) {
    stream << (map[index] ? '1' : '0');
  }
  return stream;
}

using OverflowPipe =
    sycl::ext::intel::pipe<class OverflowPipeId, OverflowBits, PIPELINE_DEPTH>;

void start_kernels(sycl::queue &q, size_t count) {
  q.submit([&](auto &h) {
    auto out = sycl::stream(4096, 1024, h);

    h.template single_task<class ComputeKernel>([=]() {
      for (auto index = size_t{0}; index < count; ++index) {
        // 1. Read line from input.
        auto input = InputPipe::read();

        out << "Block #" << index << "\n";
        out << "input: ";
        for (auto c : input) {
          out << c;
        }
        out << "\n";

        // 2. Build bitmaps for string and odd quotes.
        auto bitmaps = build_bitmaps(input);
        out << "strng: " << bitmaps.is_string << "\n"
            << "ovstr: " << bitmaps.overflows.is_string << "\n"
            << "ovodd: " << bitmaps.overflows.is_odd << "\n";
      }

      // 3. Read overflow bits and modify own bitmaps.
      // TODO: Simply flipping our bitmaps does not work here, as a string like
      // ("\|\" other) would classify other as not string, even though the
      // backslash was escaped in the previous block.
      auto last_overflow = OverflowPipe::read();

      // 4. Push out to next step.
    });
  });
}

Bitmaps build_bitmaps(const CacheLine &cache_line) {
  auto bitmaps = Bitmaps{};
  auto is_string = false;
  auto is_odd = false;
  for (auto byte_index = size_t{0}; byte_index < CACHE_LINE_SIZE;
       ++byte_index) {
    if (!is_odd && cache_line[byte_index] == '\"') {
      is_string = !is_string;
    } else if (is_string && cache_line[byte_index] == '\\') {
      is_odd = !is_odd;
    }
    bitmaps.is_string[byte_index] = is_string;
  }
  bitmaps.overflows = OverflowBits{.is_string = is_string, .is_odd = is_odd};
  return bitmaps;
}

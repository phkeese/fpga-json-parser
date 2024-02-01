#include <bitset>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

// oneAPI headers
#include "bitmaps.hpp"
#include "definitions.hpp"
#include "exception_handler.hpp"
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Functions
std::vector<std::string_view> find_strings(sycl::queue &q, std::string input);

void start_kernels(sycl::queue &q, const size_t count);

Bitmaps build_bitmaps(const CacheLine &cache_line, const Overflows last_overflow);

// Pipes

// Host -> Bitmap computation.
using InputPipe = sycl::ext::intel::experimental::pipe<class InputPipeID, CacheLine, PIPELINE_DEPTH>;

// Host -> String filter.
using CachelineToStringFilterPipe =
	sycl::ext::intel::experimental::pipe<class CachelineToStringFilterPipeID, CacheLine, PIPELINE_DEPTH>;

// Bitmap computation -> String filter (device).
using BitmapsToStringFilterPipe = sycl::ext::intel::pipe<class BitmapsToStringFilterPipeID, Bitmaps, PIPELINE_DEPTH>;

using CharOutputPipe = sycl::ext::intel::experimental::pipe<class CharOutputPipeID, char, PIPELINE_DEPTH>;

// Main function
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

		sycl::queue q(selector, fpga_tools::exception_handler, sycl::property::queue::enable_profiling{});

		auto device = q.get_device();

		std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;

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

// Function definitions
std::vector<std::string_view> find_strings(sycl::queue &q, std::string input) {
	// Append whitespaces to input to make it a multiple of CACHE_LINE_SIZE.
	auto whitespaces_to_append = CACHE_LINE_SIZE - (input.size() % CACHE_LINE_SIZE);
	input.append(whitespaces_to_append, ' ');

	auto cache_line_count = input.size() / CACHE_LINE_SIZE;
	start_kernels(q, cache_line_count);

	// Send data to device.
	auto index = size_t{0};
	for (; index < cache_line_count; ++index) {
		auto begin = input.begin() + index * CACHE_LINE_SIZE;
		auto end = input.begin() + (index + 1) * CACHE_LINE_SIZE;
		auto line = CacheLine{};
		std::copy(begin, end, line.begin());
		InputPipe::write(q, line);
		CachelineToStringFilterPipe::write(q, line);
	}

	return std::vector<std::string_view>();
}

const sycl::stream &operator<<(const sycl::stream &stream, const Bitmap &map) {
	for (auto index = size_t{0}; index < CACHE_LINE_SIZE; ++index) {
		stream << (map[index] ? '1' : '0');
	}
	return stream;
}

void start_kernels(sycl::queue &q, const size_t count) {
	compute_bitmaps<InputPipe, BitmapsToStringFilterPipe>(q, count);
}

void start_string_filter(sycl::queue &q, const size_t count) {
	std::vector<char> output;
	q.submit([&](auto &h) {
		h.template single_task<class StringFilterKernel>([=]() {
			for (auto index = size_t{0}; index < count; ++index) {
				auto bitmaps = BitmapsToStringFilterPipe::read();
				auto input = CachelineToStringFilterPipe::read();

				for (auto byte_index = size_t{0}; byte_index < CACHE_LINE_SIZE; ++byte_index) {
					const auto c = input[byte_index];
					if (bitmaps.is_string[byte_index]) {
						if ((c != '\"' && c != '\\') || bitmaps.is_escaped[byte_index]) {
							CharOutputPipe::write(c);
						}
					}
				}
			}
		});
	});
}
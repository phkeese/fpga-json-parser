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

/**
 * Write cache lines of data to a pipe.
 * @tparam WritePipe Pipe to write cache lines to.
 * @param input Data to write.
 * @return Number of cache lines to expect.
 */
template <class WritePipe> size_t write_input(sycl::queue &q, const std::string &input);

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

		constexpr auto input = R"("k":"value", "k\"y": "\"", "key": "unescaped\"" )";

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

using DebugBitmapsPipe = sycl::ext::intel::experimental::pipe<class DebugBitmapsPipeId, Bitmaps, PIPELINE_DEPTH>;

// Function definitions
std::vector<std::string_view> find_strings(sycl::queue &q, std::string input) {
	auto cache_line_count = write_input<InputPipe>(q, input);
	compute_bitmaps<InputPipe, DebugBitmapsPipe>(q, cache_line_count);
	for (auto i = 0ull; i < cache_line_count; ++i) {
		auto bitmaps = DebugBitmapsPipe::read(q);
		auto actual_state = OverflowState::None;

		std::cout << "Input: ";
		for (auto c : bitmaps.input) {
			std::cout << c;
		}
		std::cout << "\n";

		auto string_bits = bitmaps.is_string.to_string();
		std::reverse(string_bits.begin(), string_bits.end());
		auto escaped_bits = bitmaps.is_escaped.to_string();
		std::reverse(escaped_bits.begin(), escaped_bits.end());

		std::cout << "string:" << string_bits << "\n"
				  << "escapd:" << escaped_bits << "\n"
				  << "state: ";
		print(std::cout, bitmaps.overflow_state);
		std::cout << "\n";
	}

	q.wait();

	return std::vector<std::string_view>();
}

template <class WritePipe> size_t write_input(sycl::queue &q, const std::string &input) {
	assert(input.size() % CACHE_LINE_SIZE == 0);
	const auto line_count = input.size() / CACHE_LINE_SIZE;
	auto input_buffer = sycl::buffer{input};
	q.submit([&](auto &h) {
		auto input_accessor = sycl::accessor{input_buffer, h, sycl::read_only};
		h.template single_task([=]() {
			for (auto line_index = size_t{0}; line_index < line_count; ++line_index) {
				auto begin = input_accessor.begin() + line_index * CACHE_LINE_SIZE;
				auto end = input_accessor.begin() + (line_index + 1) * CACHE_LINE_SIZE;
				auto line = CacheLine{};
				std::copy(begin, end, line.begin());
				WritePipe::write(line);
			}
		});
	});
	return line_count;
}

// void start_string_filter(sycl::queue &q, const size_t count) {
//	std::vector<char> output;
//	q.submit([&](auto &h) {
//		h.template single_task<class StringFilterKernel>([=]() {
//			for (auto index = size_t{0}; index < count; ++index) {
//				auto bitmaps = BitmapsToStringFilterPipe::read();
//				auto input = CachelineToStringFilterPipe::read();
//
//				for (auto byte_index = size_t{0}; byte_index < CACHE_LINE_SIZE; ++byte_index) {
//					const auto c = input[byte_index];
//					if (bitmaps.is_string[byte_index]) {
//						if ((c != '\"' && c != '\\') || bitmaps.is_escaped[byte_index]) {
//							CharOutputPipe::write(c);
//						}
//					}
//				}
//			}
//		});
//	});
// }
#include <bitset>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

// oneAPI headers
#include "bitmaps.hpp"
#include "definitions.hpp"
#include "exception_handler.hpp"
#include "json_parser.hpp"
#include "string_filter.hpp"
#include "tape_builder.hpp"
#include "taped_json.hpp"
#include <fstream>
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Functions
void find_strings(sycl::queue &q, std::string input);

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

		auto input = std::string{
			R"({"k":"value", "k\"y": "\"",   "key": "unescaped\"", "thisisareallylongstringitinvolvesmultiplecachelines": "blub\nmore"})"};
		if (argc > 1) {
			const auto filename = argv[1];
			std::ifstream file(filename);
			if (!file.is_open()) {
				std::cerr << "Could not open file: " << filename << std::endl;
				return EXIT_FAILURE;
			}
			input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
		}

		find_strings(q, input);

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

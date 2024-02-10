#include <benchmark/benchmark.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "exception_handler.hpp"
#include "json_parser.hpp"
#include "taped_json.hpp"

constexpr auto JSON_PATH = "../data/processed";

std::vector<std::string> getAllFilenames(const std::string &folderPath) {
	std::vector<std::string> filenames;
	for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
		if (entry.is_regular_file()) {
			filenames.push_back("../data/processed/" + entry.path().filename().string());
		}
	}
	return filenames;
}

std::pair<uint64_t, uint64_t> count_objects_and_arrays(const TapedJson &json) { return {0, 0}; }

static void BM_SomeFunction(benchmark::State &state) {
	// Perform setup here
	for (auto _ : state) {
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
		auto filenames = getAllFilenames(JSON_PATH);

		for (const auto &filename : filenames) {
			// std::cout << filename << ":" << std::endl;
			std::ifstream file(filename);
			const auto input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
			const auto json = parse(q, input);
			// std::cout << json.max_depth() << std::endl;
			// break;
		}
	}
}
// Register the function as a benchmark
BENCHMARK(BM_SomeFunction);
// Run the benchmark
BENCHMARK_MAIN();

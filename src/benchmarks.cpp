#include <benchmark/benchmark.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "exception_handler.hpp"
#include "json_parser.hpp"
#include "simdjson/simdjson.h"
#include "taped_json.hpp"

using namespace simdjson;

constexpr auto JSON_PATH = "../data/processed";
// constexpr auto JSON_PATH = "../data/few";

std::vector<std::string> getAllFilenames(const std::string &folderPath) {
	std::vector<std::string> filenames;

	DIR *dpdf;
	struct dirent *epdf;

	dpdf = opendir(JSON_PATH);
	if (dpdf != NULL) {
		while (epdf = readdir(dpdf)) {
			// printf("Filename: %s", epdf->d_name);
			if (epdf->d_name[0] != '.' && epdf->d_name[0] != '..') {
				filenames.push_back(folderPath + "/" + epdf->d_name);
			}
		}
	}
	closedir(dpdf);

	// for (const auto &filename : filenames) {
	// 	std::cout << filename << std::endl;
	// }

	return filenames;
}

std::pair<uint64_t, uint64_t> count_objects_and_arrays(const TapedJson &json) { return {0, 0}; }

static void MAX_DEPTH_FPGA(benchmark::State &state) {
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
			std::cout << json.max_depth() << std::endl;
		}
	}
}

static void ONLY_PARSE_FPGA(benchmark::State &state) {
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
		}
	}
}

static void COUNT_STRING_LENGTHS_FPGA(benchmark::State &state) {
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
			std::cout << json.count_string_lengths() << std::endl;
		}
	}
}

// static void asf(benchmark::State &state) {
// 	auto parser = ondemand::parser{};

// 	auto filenames = getAllFilenames(JSON_PATH);

// 	for (const auto &filename : filenames) {
// 		const auto json = padded_string::load(filename);
// 		ondemand::document tweets = parser.iterate(json);
// 		std::cout << uint64_t(tweets["search_metadata"]["count"]) << " results." << std::endl;
// 	}
// }

// Register the function as a benchmark
BENCHMARK(MAX_DEPTH_FPGA);
BENCHMARK(ONLY_PARSE_FPGA);
BENCHMARK(COUNT_STRING_LENGTHS_FPGA);

// Run the benchmark
BENCHMARK_MAIN();

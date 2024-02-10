#include <benchmark/benchmark.h>

#include "exception_handler.hpp"
#include "json_parser.hpp"
#include "taped_json.hpp"
#include <fstream>
#include <iostream>

static sycl::queue setup_queue() {
#if FPGA_SIMULATOR
	auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
	auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
	auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
	sycl::queue q(selector, fpga_tools::exception_handler, sycl::property::queue::enable_profiling{});
	return q;
}

static void MAX_DEPTH_FPGA(benchmark::State &state, const std::string &filename) {
	// Perform setup here
	auto q = setup_queue();
	std::ifstream file(filename);
	const auto input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

	for (auto _ : state) {
		const auto json = parse(q, input);
		auto max_depth = json.max_depth();
		(void)max_depth;
		//		std::cout << max_depth << std::endl;
	}
}

static void ONLY_PARSE_FPGA(benchmark::State &state, const std::string &filename) {
	// Perform setup here
	auto q = setup_queue();
	std::ifstream file(filename);
	const auto input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

	for (auto _ : state) {
		const auto json = parse(q, input);
		(void)json;
	}
}

static void COUNT_STRING_LENGTHS_FPGA(benchmark::State &state, const std::string &filename) {
	// Perform setup here
	auto q = setup_queue();
	std::ifstream file(filename);
	const auto input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

	for (auto _ : state) {
		const auto json = parse(q, input);
		auto count = json.count_string_lengths();
		(void)count;
		//		std::cout << count << std::endl;
	}
}

void register_fpga_benchmarks_for(const std::string &dirname, const std::string &filename) {
	// Register the function as a benchmark
	benchmark::RegisterBenchmark("fpga::max_depth::" + filename, MAX_DEPTH_FPGA, dirname + filename);
	benchmark::RegisterBenchmark("fpga::parse::" + filename, ONLY_PARSE_FPGA, dirname + filename);
	benchmark::RegisterBenchmark("fpga::string_lengths::" + filename, COUNT_STRING_LENGTHS_FPGA, dirname + filename);
}
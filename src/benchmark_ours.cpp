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

	for (auto _ : state) {
		std::ifstream file(filename);
		const auto input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
		const auto json = parse(q, input);
		auto max_depth = json.max_depth();
		(void)max_depth;
	}
}

static void ONLY_PARSE_FPGA(benchmark::State &state, const std::string &filename) {
	// Perform setup here
	auto q = setup_queue();

	for (auto _ : state) {
		std::ifstream file(filename);
		const auto input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
		const auto json = parse(q, input);
		(void)json;
	}
}

static void COUNT_STRING_LENGTHS_FPGA(benchmark::State &state, const std::string &filename) {
	// Perform setup here
	auto q = setup_queue();

	for (auto _ : state) {
		std::ifstream file(filename);
		const auto input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
		const auto json = parse(q, input);
		(void)json.count_string_lengths();
	}
}

void register_fpga_benchmarks_for(const std::string &filename) {
	// Register the function as a benchmark
	benchmark::RegisterBenchmark("MAX_DEPTH_FPGA_" + filename, MAX_DEPTH_FPGA, filename);
	benchmark::RegisterBenchmark("ONLY_PARSE_FPGA_" + filename, ONLY_PARSE_FPGA, filename);
	benchmark::RegisterBenchmark("COUNT_STRING_LENGTHS_FPGA_" + filename, COUNT_STRING_LENGTHS_FPGA, filename);
}
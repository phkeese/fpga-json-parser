
#include "benchmark/benchmark.h"
static void MAX_DEPTH_SIMDJSON(benchmark::State &state, const std::string &filename) {
	for (auto _ : state) {
		// Todo
	}
}

static void ONLY_PARSE_SIMDJSON(benchmark::State &state, const std::string &filename) {
	for (auto _ : state) {
		// Todo
	}
}

static void COUNT_STRING_LENGTHS_SIMDJSON(benchmark::State &state, const std::string &filename) {
	for (auto _ : state) {
		// Todo
	}
}

void register_simdjson_benchmarks_for(const std::string &dirname, const std::string &filename) {
	// Register the function as a benchmark
	benchmark::RegisterBenchmark("simdjson::max_depth::" + filename, MAX_DEPTH_SIMDJSON, dirname + filename);
	benchmark::RegisterBenchmark("simdjson::parse::" + filename, ONLY_PARSE_SIMDJSON, dirname + filename);
	benchmark::RegisterBenchmark("simdjson::string_lengths::" + filename, COUNT_STRING_LENGTHS_SIMDJSON,
								 dirname + filename);
}
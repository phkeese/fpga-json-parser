#include <benchmark/benchmark.h>
#include <iostream>

static void BM_SomeFunction(benchmark::State &state) {
	// Perform setup here
	for (auto _ : state) {
		// This code gets timed
		int t;
		(void)t;
	}
}
// Register the function as a benchmark
BENCHMARK(BM_SomeFunction);
// Run the benchmark
BENCHMARK_MAIN();

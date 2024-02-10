#include <benchmark/benchmark.h>
#include <dirent.h>
#include <iostream>

constexpr auto JSON_PATH = "../data/processed/";

extern void register_fpga_benchmarks_for(const std::string &dirname, const std::string &filename);
extern void register_simdjson_benchmarks_for(const std::string &dirname, const std::string &filename);

std::vector<std::string> getAllFilenames(const std::string &folderPath) {
	std::vector<std::string> filenames;

	DIR *dpdf;
	struct dirent *epdf;

	dpdf = opendir(JSON_PATH);
	if (dpdf) {
		while ((epdf = readdir(dpdf))) {
			if (epdf->d_name[0] != '.') {
				filenames.emplace_back(epdf->d_name);
			}
		}
	}
	closedir(dpdf);

	return filenames;
}

// Implement the benchmark fixture.

// Run the benchmark
int main(int argc, char **argv) {
	// for each filename, register a new benchmark.
	auto filenames = getAllFilenames(JSON_PATH);
	for (const auto &filename : filenames) {
		std::cout << "Registering benchmarks for " << filename << std::endl;
		register_fpga_benchmarks_for(JSON_PATH, filename);
		register_simdjson_benchmarks_for(JSON_PATH, filename);
	}
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
}
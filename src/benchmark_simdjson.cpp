#include "simdjson/simdjson.h"
#include <benchmark/benchmark.h>

using namespace simdjson;
using namespace simdjson::ondemand;

static size_t value_depth(simdjson_result<ondemand::value> value);
static size_t array_depth(simdjson_result<ondemand::array> array);
static size_t object_depth(simdjson_result<ondemand::object> object);

static size_t value_depth(simdjson_result<ondemand::value> value) {
	switch (value.type()) {
	case json_type::array:
		return array_depth(value.get_array());
	case json_type::object:
		return object_depth(value.get_object());
	default:
		return 0;
	}
}

static size_t array_depth(simdjson_result<ondemand::array> array) {
	size_t max_depth = 0;
	for (auto element : array) {
		max_depth = std::max(max_depth, value_depth(element));
	}
	return max_depth + 1;
}

static size_t object_depth(simdjson_result<ondemand::object> object) {
	size_t max_depth = 0;
	for (auto field : object) {
		max_depth = std::max(max_depth, value_depth(field.value()));
	}
	return max_depth + 1;
}

static void MAX_DEPTH_SIMDJSON(benchmark::State &state, const std::string &filename) {
	const auto json = simdjson::padded_string::load(filename);

	for (auto _ : state) {
		auto parser = simdjson::ondemand::parser{};
		auto document = parser.iterate(json);
		auto value = document.get_value();
		auto depth = value_depth(value);
		(void)depth;
		//		std::cout << depth << std::endl;
	}
}

static void ONLY_PARSE_SIMDJSON(benchmark::State &state, const std::string &filename) {
	const auto json = simdjson::padded_string::load(filename);
	for (auto _ : state) {
		auto parser = simdjson::ondemand::parser{};
		auto document = parser.iterate(json);
		auto value = document.get_value();
		(void)value;
	}
}

static size_t count_string_lengths(simdjson_result<ondemand::value> value);
static size_t count_string_lengths(simdjson_result<ondemand::array> array);
static size_t count_string_lengths(simdjson_result<ondemand::object> object);

static size_t count_string_lengths(simdjson_result<ondemand::value> value) {
	switch (value.type()) {
	case json_type::array:
		return count_string_lengths(value.get_array());
	case json_type::object:
		return count_string_lengths(value.get_object());
	case json_type::string:
		return value.get_string().value().size();
	default:
		return 0;
	}
}

static size_t count_string_lengths(simdjson_result<ondemand::array> array) {
	size_t count = 0;
	for (auto element : array) {
		count += count_string_lengths(element);
	}
	return count;
}

static size_t count_string_lengths(simdjson_result<ondemand::object> object) {
	size_t count = 0;
	for (auto field : object) {
		count += field.unescaped_key().value().size();
		count += count_string_lengths(field.value());
	}
	return count;
}

static void COUNT_STRING_LENGTHS_SIMDJSON(benchmark::State &state, const std::string &filename) {
	const auto json = simdjson::padded_string::load(filename);
	for (auto _ : state) {
		auto parser = simdjson::ondemand::parser{};
		auto document = parser.iterate(json);
		auto value = document.get_value();
		auto count = count_string_lengths(value);
		(void)count;
		//		std::cout << count << std::endl;
	}
}

void register_simdjson_benchmarks_for(const std::string &dirname, const std::string &filename) {
	// Register the function as a benchmark
	benchmark::RegisterBenchmark("simdjson::max_depth::" + filename, MAX_DEPTH_SIMDJSON, dirname + filename);
	benchmark::RegisterBenchmark("simdjson::parse::" + filename, ONLY_PARSE_SIMDJSON, dirname + filename);
	benchmark::RegisterBenchmark("simdjson::string_lengths::" + filename, COUNT_STRING_LENGTHS_SIMDJSON,
								 dirname + filename);
}
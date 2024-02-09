#pragma once

#include "definitions.hpp"
#include "tape_builder.hpp"
#include "taped_json.hpp"

template <class WritePipe> size_t write_input(sycl::queue &q, std::string &input) {
	// Check if input size is divisible by CACHE_LINE_SIZE
	if (input.size() % CACHE_LINE_SIZE != 0) {
		// Calculate the number of whitespaces to add
		const auto whitespace_count = CACHE_LINE_SIZE - (input.size() % CACHE_LINE_SIZE);
		// Append whitespaces to the input
		input.append(whitespace_count, ' ');
	}
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

TapedJson parse(sycl::queue &q, std::string &input) {
	const auto cache_line_count = write_input<InputPipe>(q, input);
	const auto taped_json = find_strings(q, cache_line_count);

	return taped_json;
}
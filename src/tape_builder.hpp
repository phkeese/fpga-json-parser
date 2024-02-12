#pragma once

#include "definitions.hpp"
#include "string_filter.hpp"
#include "taped_json.hpp"

template <typename Id, typename OutPipe>
std::pair<sycl::event, OutputCacheLine *> submit_consumer(sycl::queue &q, const size_t cache_line_count) {

	OutputCacheLine *output_cache_lines;
	if ((output_cache_lines = sycl::malloc_device<OutputCacheLine>(cache_line_count, q)) == nullptr) {
		std::cerr << "ERROR: could not allocate space for 'in'\n";
		std::terminate();
	}

	const auto consumer_event = q.submit([&](auto &h) {
		h.template single_task<Id>([=]() {
			for (auto index = size_t{0}; index < cache_line_count; ++index) {
				const auto output = OutPipe::read();
				output_cache_lines[index] = output;
			}
		});
	});

	return {consumer_event, output_cache_lines};
}

TapedJson build_tape(const size_t cache_line_count, const OutputCacheLine *output_cache_lines) {
	auto strings = std::vector<std::string>{};
	auto tape = std::vector<Token>{};
	auto had_overflow = false;
	for (auto index = size_t{0}; index < cache_line_count; ++index) {
		const auto &chars = output_cache_lines[index].line;
		const auto &lengths = output_cache_lines[index].string_lengths;

		const auto string_count = lengths[0];
		auto char_index = size_t{0};

		auto string_index = size_t{0};
		if (had_overflow && lengths[CACHE_LINE_SIZE - 2] == 1) {
			const auto string_length = lengths[++string_index];
			// std::cout << "before: " << strings.back();
			strings.back().append(chars.begin(), chars.begin() + string_length);
			// std::cout << " after: " << strings.back() << std::endl;
			char_index += string_length;
		}

		for (; string_index < string_count; ++string_index) {
			const auto string_length = lengths[string_index + 1];
			strings.emplace_back(std::string{chars.begin() + char_index, chars.begin() + char_index + string_length});
			char_index += string_length;
		}

		if (lengths[CACHE_LINE_SIZE - 1] == 1) {
			had_overflow = true;
		} else {
			had_overflow = false;
		}

		// Get the output from the tokenizer.
		const auto &tokens = output_cache_lines[index].tokens;
		for (auto token_index = size_t{0}; token_index < CACHE_LINE_SIZE; ++token_index) {
			const auto token = static_cast<Token>(tokens[token_index]);

			if (token == Token::EndOfTokens) {
				break;
			}

			tape.push_back(token);
		}
	}

	const auto taped_json = TapedJson{std::move(tape), std::move(strings)};

	return taped_json;
}
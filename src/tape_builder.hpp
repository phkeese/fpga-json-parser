#pragma once

#include "definitions.hpp"
#include "string_filter.hpp"
#include "taped_json.hpp"

template <typename OutPipe> TapedJson build_tape(sycl::queue &q, const size_t cache_line_count) {
	std::vector<std::string> strings;
	auto tape = std::vector<Token>{};
	auto had_overflow = false;
	for (auto index = size_t{0}; index < cache_line_count; ++index) {
		// Get the output from the string filter.
		const auto output = OutPipe::read(q);

		const auto &chars = output.line;
		const auto &lengths = output.string_lengths;

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
		const auto &tokens = output.tokens;
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
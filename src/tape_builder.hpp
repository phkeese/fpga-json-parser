#pragma once

#include "definitions.hpp"
#include "taped_json.hpp"

using DebugBitmapsPipe = sycl::ext::intel::pipe<class DebugBitmapsPipeId, Bitmaps, PIPELINE_DEPTH>;

void find_strings(sycl::queue &q, std::string input) {
	auto cache_line_count = write_input<InputPipe>(q, input);

	compute_bitmaps<InputPipe, DebugBitmapsPipe, TokenizedCachelinesToStringFilterPipe>(q, cache_line_count);
	start_string_filter<TokenizedCachelinesToStringFilterPipe, OutputCacheLinePipe>(q, cache_line_count);

	auto output_bitmaps = std::vector<Bitmaps>(cache_line_count);
	auto output_buffer = sycl::buffer{output_bitmaps};

	q.submit([&](auto &h) {
		auto output_accessor = sycl::accessor{output_buffer, h, sycl::write_only};
		h.template single_task<class ReadKernel>([=]() {
			for (auto index = size_t{0}; index < cache_line_count; ++index) {
				output_accessor[index] = DebugBitmapsPipe::read();
			}
		});
	});
	q.wait();

	// for (auto &bitmaps : output_bitmaps) {
	// 	std::cout << "Input: ";
	// 	for (auto c : bitmaps.input) {
	// 		if (std::isprint(c)) {
	// 			std::cout << c;
	// 		} else {
	// 			std::cout << ".";
	// 		}
	// 	}
	// 	std::cout << "\n";

	// 	auto string_bits = bitmaps.is_string.to_string();
	// 	std::reverse(string_bits.begin(), string_bits.end());
	// 	auto escaped_bits = bitmaps.is_escaped.to_string();
	// 	std::reverse(escaped_bits.begin(), escaped_bits.end());

	// 	std::cout << "string:" << string_bits << "\n"
	// 			  << "escapd:" << escaped_bits << "\n"
	// 			  << "state: ";
	// 	print(std::cout, bitmaps.overflow_state);
	// 	std::cout << "\n";
	// }

	std::vector<std::string> strings;
	auto tape = std::vector<Token>{};
	auto had_overflow = false;
	for (auto index = size_t{0}; index < cache_line_count; ++index) {
		// Get the output from the string filter.
		const auto output = OutputCacheLinePipe::read(q);

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
	// std::cout << "taped_json.print_tokes():" << std::endl;
	// taped_json.print_tokes();
	// std::cout << "\n taped_json.print_strings():" << std::endl;
	// taped_json.print_strings();
	std::cout << "\n taped_json.print_tape():" << std::endl;
	taped_json.print_tape();
}
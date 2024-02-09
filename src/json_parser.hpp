#pragma once

#include "bitmaps.hpp"
#include "definitions.hpp"
#include "string_filter.hpp"
#include "tape_builder.hpp"
#include "taped_json.hpp"

// declare kernel and pipe names at the global scope to reduce name mangling

// Host -> Bitmap computation.
class ProducerId;
class InPipeId;
using InPipe = sycl::ext::intel::experimental::pipe<InPipeId, CacheLine, PIPELINE_DEPTH>;

// Bitmap computation -> String filter (both device).
class TokenizerId;
class TokenizerToStringFilterPipeId;
using TokenizerToStringFilterPipe =
	sycl::ext::intel::pipe<TokenizerToStringFilterPipeId, TokenizedCacheLine, PIPELINE_DEPTH>;

// String filter -> Tape Builder (Host).
class StringFilterId;
class OutPipeId;
using OutPipe = sycl::ext::intel::experimental::pipe<OutPipeId, OutputCacheLine, PIPELINE_DEPTH>;

// class DebugPipeId;
// using DebugPipe = sycl::ext::intel::experimental::pipe<DebugPipeId, Bitmaps, PIPELINE_DEPTH>;

template <typename Id, typename InPipe> sycl::event submit_producer(sycl::queue &q, const std::string &input) {
	const auto input_size = input.size();

	// Check if input size is divisible by CACHE_LINE_SIZE
	const bool underfull_cache_line = input_size % CACHE_LINE_SIZE != 0;

	const auto cache_line_count = input_size / CACHE_LINE_SIZE;
	auto input_buffer = sycl::buffer{input};

	const auto producer_event = q.submit([&](auto &h) {
		const auto input_accessor = sycl::accessor{input_buffer, h, sycl::read_only};

		h.template single_task<Id>([=]() {
			for (auto index = size_t{0}; index < cache_line_count; ++index) {
				const auto begin = input_accessor.begin() + index * CACHE_LINE_SIZE;
				const auto end = input_accessor.begin() + (index + 1) * CACHE_LINE_SIZE;
				auto line = CacheLine{};
				std::copy(begin, end, line.begin());
				InPipe::write(line);
			}

			if (underfull_cache_line) {
				const auto begin = input_accessor.begin() + cache_line_count * CACHE_LINE_SIZE;
				const auto end = input_accessor.begin() + input_size;
				auto line = CacheLine{};
				std::copy(begin, end, line.begin());
				std::fill(line.begin() + (input_size % CACHE_LINE_SIZE), line.end(), ' ');
				InPipe::write(line);
			}
		});
	});

	return producer_event;
}

TapedJson parse(sycl::queue &q, std::string &input) {
	const auto cache_line_count =
		input.size() % CACHE_LINE_SIZE == 0 ? input.size() / CACHE_LINE_SIZE : input.size() / CACHE_LINE_SIZE + 1;

	const auto producer_event = submit_producer<ProducerId, InPipe>(q, input);
	const auto tokenizer_event =
		submit_tokenizer<TokenizerId, InPipe, TokenizerToStringFilterPipe>(q, cache_line_count);

	// const auto debug_tokenizer_event = submit_tokenizer<TokenizerId, InPipe, TokenizerToStringFilterPipe,
	// DebugPipe>(q, cache_line_count);

	const auto string_filter_event =
		submit_string_filter<StringFilterId, TokenizerToStringFilterPipe, OutPipe>(q, cache_line_count);

	const auto taped_json = build_tape<OutPipe>(q, cache_line_count);

	return taped_json;
}

///////////////////////////////////////
// Comment in for debugging purposes //
///////////////////////////////////////
/*
template <typename Id, typename DebugPipe> void debug_tokenizer(sycl::queue &q, const size_t cache_line_count) {
	auto output_bitmaps = std::vector<Bitmaps>(cache_line_count);
	auto output_buffer = sycl::buffer{output_bitmaps};

	q.submit([&](auto &h) {
		auto output_accessor = sycl::accessor{output_buffer, h, sycl::write_only};
		h.template single_task<Id>([=]() {
			for (auto index = size_t{0}; index < cache_line_count; ++index) {
				output_accessor[index] = DebugPipe::read();
			}
		});
	});
	q.wait();

	for (auto &bitmaps : output_bitmaps) {
		std::cout << "Input: ";
		for (auto c : bitmaps.input) {
			if (std::isprint(c)) {
				std::cout << c;
			} else {
				std::cout << ".";
			}
		}
		std::cout << "\n";

		auto string_bits = bitmaps.is_string.to_string();
		std::reverse(string_bits.begin(), string_bits.end());
		auto escaped_bits = bitmaps.is_escaped.to_string();
		std::reverse(escaped_bits.begin(), escaped_bits.end());

		std::cout << "string:" << string_bits << "\n"
				  << "escapd:" << escaped_bits << "\n"
				  << "state: ";
		print(std::cout, bitmaps.overflow_state);
		std::cout << "\n";
	}
}*/

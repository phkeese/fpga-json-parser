#pragma once

#include <sycl/sycl.hpp>

#include "definitions.hpp"
#include "string_filter.hpp"
#include "tape_builder.hpp"
#include "taped_json.hpp"
#include "tokenizer.hpp"

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

class ConsumerId;

template <typename Id, typename InPipe>
std::pair<sycl::event, size_t> submit_producer(sycl::queue &q, const std::string &input) {
	const auto input_size = input.size();

	char *in;
	if ((in = sycl::malloc_device<char>(input_size, q)) == nullptr) {
		std::cerr << "ERROR: could not allocate space for 'in'\n";
		std::terminate();
	}

	q.memcpy(in, input.data(), input_size * sizeof(char)).wait();

	// Check if input size is divisible by CACHE_LINE_SIZE
	const bool underfull_cache_line = input_size % CACHE_LINE_SIZE != 0;

	const auto cache_line_count = input_size / CACHE_LINE_SIZE;
	// auto input_buffer = sycl::buffer{input};

	const auto producer_event = q.submit([&](auto &h) {
		// const auto input_accessor = sycl::accessor{input_buffer, h, sycl::read_only};

		h.template single_task<Id>([=]() {
			for (auto index = size_t{0}; index < cache_line_count; ++index) {
				const auto *begin = in + index * CACHE_LINE_SIZE;
				const auto *end = in + (index + 1) * CACHE_LINE_SIZE;
				auto line = CacheLine{};
				std::copy(begin, end, line.begin());
				InPipe::write(line);
			}

			if (underfull_cache_line) {
				const auto *begin = in + cache_line_count * CACHE_LINE_SIZE;
				const auto *end = in + input_size;
				auto line = CacheLine{};
				std::copy(begin, end, line.begin());
				std::fill(line.begin() + (input_size % CACHE_LINE_SIZE), line.end(), ' ');
				InPipe::write(line);
			}
		});
	});

	const auto final_cache_line_count = underfull_cache_line ? cache_line_count + 1 : cache_line_count;
	return {producer_event, final_cache_line_count};
}

TapedJson parse(sycl::queue &q, const std::string &input) {
	const auto [producer_event, cache_line_count] = submit_producer<ProducerId, InPipe>(q, input);

	const auto tokenizer_event =
		submit_tokenizer<TokenizerId, InPipe, TokenizerToStringFilterPipe>(q, cache_line_count);

	const auto string_filter_event =
		submit_string_filter<StringFilterId, TokenizerToStringFilterPipe, OutPipe>(q, cache_line_count);

	auto output_cache_lines = std::vector<OutputCacheLine>{};
	const auto consumer_event = submit_consumer<ConsumerId, OutPipe>(q, cache_line_count, output_cache_lines);

	const auto taped_json = build_tape(cache_line_count, output_cache_lines);
	return taped_json;
}

#pragma once

#include <thread>

#include <sycl/sycl.hpp>

#include "definitions.hpp"
#include "string_filter.hpp"
#include "tape_builder.hpp"
#include "taped_json.hpp"
#include "tokenizer.hpp"

// Host -> Bitmap computation.
class ProducerId;
class InPipeId;
using InPipe = sycl::ext::intel::pipe<InPipeId, CacheLine, PIPELINE_DEPTH>;

// Bitmap computation -> String filter (both device).
class TokenizerId;
class TokenizerToStringFilterPipeId;
using TokenizerToStringFilterPipe =
	sycl::ext::intel::pipe<TokenizerToStringFilterPipeId, TokenizedCacheLine, PIPELINE_DEPTH>;

// String filter -> Tape Builder (Host).
class StringFilterId;
class OutPipeId;
using OutPipe = sycl::ext::intel::pipe<OutPipeId, OutputCacheLine, PIPELINE_DEPTH>;

class ConsumerId;

template <typename Id, typename InPipe> size_t submit_producer(sycl::queue &q, const std::string &input) {
	const auto input_size = input.size();
	const auto cache_line_count = input_size / CACHE_LINE_SIZE;
	const bool underfull_cache_line = input_size % CACHE_LINE_SIZE != 0;

	auto produce_cachelines = [](sycl::queue &q, const std::string &input) {
		const auto input_size = input.size();
		const auto cache_line_count = input_size / CACHE_LINE_SIZE;

		// Check if input size is divisible by CACHE_LINE_SIZE
		const bool underfull_cache_line = input_size % CACHE_LINE_SIZE != 0;

		for (auto index = size_t{0}; index < cache_line_count; ++index) {
			const auto begin = input.begin() + index * CACHE_LINE_SIZE;
			const auto end = input.begin() + (index + 1) * CACHE_LINE_SIZE;
			auto line = CacheLine{};
			std::copy(begin, end, line.begin());
			auto line_buffer = sycl::buffer{line};

			q.submit([&](auto &h) {
				const auto line_accessor = sycl::accessor{line_buffer, h, sycl::read_only};

				h.single_task([=]() {
					const auto begin = line_accessor.begin();
					const auto end = line_accessor.begin() + CACHE_LINE_SIZE;
					auto cache_line = CacheLine{};
					std::copy(begin, end, cache_line.begin());
					InPipe::write(cache_line);
				});
			});
		}
		if (underfull_cache_line) {
			const auto begin = input.begin() + cache_line_count * CACHE_LINE_SIZE;
			const auto end = input.begin() + input_size;
			auto line = CacheLine{};
			std::copy(begin, end, line.begin());
			std::fill(line.begin() + (input_size % CACHE_LINE_SIZE), line.end(), ' ');
			auto line_buffer = sycl::buffer{line};

			q.submit([&](auto &h) {
				const auto line_accessor = sycl::accessor{line_buffer, h, sycl::read_only};

				h.single_task([=]() {
					const auto begin = line_accessor.begin();
					const auto end = line_accessor.begin() + CACHE_LINE_SIZE;
					auto cache_line = CacheLine{};
					std::copy(begin, end, cache_line.begin());
					InPipe::write(cache_line);
				});
			});
		}
	};

	auto producer_thread = std::thread(produce_cachelines, std::ref(q), std::ref(input));
	producer_thread.detach();

	const auto final_cache_line_count = underfull_cache_line ? cache_line_count + 1 : cache_line_count;
	return final_cache_line_count;
}

TapedJson parse(sycl::queue &q, const std::string &input) {
	auto cache_line_count = submit_producer<ProducerId, InPipe>(q, input);

	const auto tokenizer_event =
		submit_tokenizer<TokenizerId, InPipe, TokenizerToStringFilterPipe>(q, cache_line_count);

	const auto string_filter_event =
		submit_string_filter<StringFilterId, TokenizerToStringFilterPipe, OutPipe>(q, cache_line_count);

	auto output_cache_lines = std::vector<OutputCacheLine>{};
	const auto consumer_event = submit_consumer<ConsumerId, OutPipe>(q, cache_line_count, output_cache_lines);

	const auto taped_json = build_tape(cache_line_count, output_cache_lines);
	return taped_json;
}

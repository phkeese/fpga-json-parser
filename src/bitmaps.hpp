#pragma once

#include "definitions.hpp"
#include "unrolled_loop.hpp"
#include <pipe_utils.hpp>
#include <string_view>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

/**
 * Compute the bitmaps for a single type of overflow on a single cache line.
 * @tparam InitialState Which overflow state to start with.
 * @param input Input to compute bitmaps for.
 * @return Bitmaps for concrete initial state and input.
 */
template <OverflowType InitialState> Bitmaps compute_bitmaps(const CacheLine &input);

template <OverflowType> class ComputeBitmapsKernel;

/**
 * Compute bitmaps by reading cache lines from input pipe and writing complete
 * bitmaps to output pipe.
 * @tparam CacheLineInputPipe Input pipe to read from.
 * @tparam BitmapsOutputPipe Output pipe to write to.
 * @param q Queue to use.
 * @param expect Number of cache lines to expect.
 */
template <typename CacheLineInputPipe, typename BitmapsOutputPipe> void compute_bitmaps(sycl::queue &q, size_t expect) {
	// 1. Read input and write to pipe for each possible overflow type.
	using OverflowPipes = fpga_tools::PipeArray<class ParallelBitmapsInId, // Easier to debug.
												CacheLine,				   // Supply each parallel computation.
												PIPELINE_DEPTH,			   // Same as everywhere.
												OverflowType::COUNT,	   // One row per possible overflow type.
												1						   // Just one column.
												>;
	q.template single_task<class SplitInputKernel>([=]() {
		for (auto index = size_t{0}; index < expect; ++index) {
			auto input = CacheLineInputPipe::read();
			OverflowPipes ::write(input);
		}
	});

	// 2. For each possible overflow type, read from corresponding pipe and compute bitmaps.
	fpga_tools::UnrolledLoop<OverflowType::COUNT>([&](auto id) {
		q.template single_task<class ComputeBitmapsKernel<id>>([=]() { auto r = OverflowPipes::PipeAt<id>::read(); });
	});
}

OverflowType encode_overflow(bool is_string, bool is_odd) {
	if (is_string) {
		return is_odd ? OverflowType::StringWithBackslash : OverflowType::String;
	} else {
		return OverflowType::None;
	}
}

template <OverflowType InitialState> Bitmaps compute_bitmaps(const CacheLine &input) {
	auto bitmaps = Bitmaps{};
	auto is_string = InitialState == OverflowType::String || InitialState == OverflowType::StringWithBackslash;
	auto is_odd = InitialState == OverflowType::StringWithBackslash;
	for (auto byte_index = size_t{0}; byte_index < CACHE_LINE_SIZE; ++byte_index) {
		bitmaps.is_escaped[byte_index] = is_odd;

		if (!is_odd && input[byte_index] == '\"') {
			is_string = !is_string;
		} else if (!is_odd && input[byte_index] == '\\') {
			is_odd = true;
		} else if (is_odd) {
			is_odd = false;
		}

		bitmaps.is_string[byte_index] = is_string;
	}

	bitmaps.overflow_type = encode_overflow(is_string, is_odd);
	return bitmaps;
}
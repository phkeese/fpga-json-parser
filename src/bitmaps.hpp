#pragma once

#include "definitions.hpp"
#include "unrolled_loop.hpp"
#include <exception>
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
Bitmaps compute_bitmaps(OverflowState state, const CacheLine &input);

class ComputeBitmapsKernel;

const sycl::stream &operator<<(const sycl::stream &stream, const Bitmap &map) {
	for (auto index = size_t{0}; index < CACHE_LINE_SIZE; ++index) {
		stream << (map[index] ? '1' : '0');
	}
	return stream;
}

/**
 * Compute bitmaps by reading cache lines from input pipe and writing complete
 * bitmaps to output pipe.
 * @tparam CacheLineInputPipe Input pipe to read from.
 * @tparam BitmapsOutputPipe Output pipe to write to.
 * @param q Queue to use.
 * @param expect Number of cache lines to expect.
 */
template <typename CacheLineInputPipe, typename BitmapsOutputPipe> void compute_bitmaps(sycl::queue &q, size_t expect) {
	using StatePipe = sycl::ext::intel::pipe<class StatePipeId, OverflowState, PIPELINE_DEPTH>;
	q.submit([&](auto &h) {
		auto out = sycl::stream(4096, 1024, h);

		h.template single_task<class ComputeBitmapsKernel>([=]() {
			for (auto line_index = size_t{0}; line_index < expect; ++line_index) {
				const auto input = CacheLineInputPipe::read();
				auto possible_bitmaps = std::array<Bitmaps, OverflowState::COUNT>{};
				for (auto state = size_t{0}; state < OverflowState::COUNT; ++state) {
					possible_bitmaps[state] = compute_bitmaps(static_cast<OverflowState>(state), input);
				}

				auto actual_state = OverflowState::None;
				if (line_index != 0) {
					actual_state = StatePipe::read();
				}
				auto actual_bitmaps = possible_bitmaps[actual_state];
				StatePipe::write(actual_bitmaps.overflow_state);
				BitmapsOutputPipe::write(actual_bitmaps);

				out << "Input: ";
				for (auto c : input) {
					out << c;
				}
				out << "\n";

				out << "string:" << actual_bitmaps.is_string << "\n"
					<< "escapd:" << actual_bitmaps.is_escaped << "\n"
					<< "state: " << OverflowStateStrings[actual_bitmaps.overflow_state] << "\n";
			}
		});
	});
}

Bitmaps compute_bitmaps(OverflowState state, const CacheLine &input) {
	auto bitmaps = Bitmaps{};
	for (auto byte_index = size_t{0}; byte_index < CACHE_LINE_SIZE; ++byte_index) {
		const auto here = input[byte_index];
		switch (state) {
		case OverflowState::None:
			if (here == '\"') {
				state = OverflowState::String;
			}
			break;
		case OverflowState::String:
			if (here == '\"') {
				state = OverflowState::None;
			} else if (here == '\\') {
				state = OverflowState::StringWithBackslash;
			}
			break;
		case OverflowState::StringWithBackslash:
			bitmaps.is_escaped[byte_index] = true;
			state = OverflowState::String;
			break;
		default:
			// Cannot throw an exception, we just hope it works.
			break;
		}
		bitmaps.is_string[byte_index] = state == OverflowState::String || state == OverflowState::StringWithBackslash;
	}
	bitmaps.overflow_state = state;
	return bitmaps;
}
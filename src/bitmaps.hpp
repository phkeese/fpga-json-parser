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
 * @return pair of Bitmaps for concrete initial state and input and the tokens inside a CacheLine
 */
std::pair<Bitmaps, CacheLine>  compute_bitmaps(OverflowState state, const CacheLine &input);

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
template <typename CacheLineInputPipe, typename BitmapsOutputPipe, typename TokenizedBitmapsToStringFilterPipe> void compute_bitmaps(sycl::queue &q, size_t expect) {
	q.template single_task<class ComputeBitmapsKernel>([=]() {
		auto actual_state = OverflowState::None;

		for (auto line_index = size_t{0}; line_index < expect; ++line_index) {
			const auto input = CacheLineInputPipe::read();

			const auto [bitmaps, tokens] = compute_bitmaps(actual_state, input);
			actual_state = bitmaps.overflow_state;

			BitmapsOutputPipe::write(bitmaps);
			TokenizedBitmapsToStringFilterPipe::write({input, bitmaps, tokens});
		}

		assert(actual_state == OverflowState::None);
	});
}

std::pair<Bitmaps, CacheLine> compute_bitmaps(OverflowState state, const CacheLine &input) {
	auto bitmaps = Bitmaps{};

	auto token_index = size_t{0};
	auto tokens = CacheLine{};

	auto emit_token = [&](Token token) {
		tokens[token_index++] = token;
	};

	bitmaps.input = input;
	for (auto byte_index = size_t{0}; byte_index < CACHE_LINE_SIZE; ++byte_index) {
		const auto here = input[byte_index];
		switch (state) {
		case OverflowState::None:
			switch (here) {
				case '{':
					emit_token(Token::ObjectBeginToken);
					break;
				case '}':
					emit_token(Token::ObjectEndToken);
					break;
				case '[':
					emit_token(Token::ArrayBeginToken);
					break;
				case ']':
					emit_token(Token::ArrayEndToken);
					break;
				case '"':
					emit_token(Token::StringToken);
					state = OverflowState::String;
					break;
				default:
					// not supposed to happen for now
					break;
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

	if (token_index != CACHE_LINE_SIZE) {
		emit_token(Token::EndOfTokens);
	}

	bitmaps.overflow_state = state;
	return {bitmaps, tokens};
}
#pragma once

#include "definitions.hpp"
#include "unrolled_loop.hpp"
#include <exception>
#include <pipe_utils.hpp>
#include <string_view>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

const sycl::stream &operator<<(const sycl::stream &stream, const Bitmap &map) {
	for (auto index = size_t{0}; index < CACHE_LINE_SIZE; ++index) {
		stream << (map[index] ? '1' : '0');
	}
	return stream;
}

/**
 * Compute the bitmaps for a single type of overflow on a single cache line.
 * @tparam InitialState Which overflow state to start with.
 * @param input Input to compute bitmaps for.
 * @return pair of Bitmaps for concrete initial state and input and the tokens inside a CacheLine
 */
std::pair<Bitmaps, CacheLine> compute_bitmaps(OverflowState state, const CacheLine &input) {
	auto bitmaps = Bitmaps{};

	auto token_index = size_t{0};
	auto tokens = CacheLine{};

	auto emit_token = [&](Token token) { tokens[token_index++] = token; };

	// bitmaps.input = input;
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

/**
 * Compute bitmaps by reading cache lines from input pipe and writing complete
 * bitmaps to output pipe.
 * @tparam CacheLineInputPipe Input pipe to read from.
 * @tparam BitmapsOutputPipe Output pipe to write to.
 * @param q Queue to use.
 * @param cache_line_count Number of cache lines to expect.
 */
template <typename Id, typename InPipe, typename OutPipe>
sycl::event submit_tokenizer(sycl::queue &q, const size_t cache_line_count) {
	const auto tokenizer_event = q.submit([&](auto &h) {
		// auto out = sycl::stream(4096, 1024, h);
		h.template single_task<Id>([=]() {
			auto last_overflow_state = OverflowState::None;

			for (auto line_index = size_t{0}; line_index < cache_line_count; ++line_index) {
				const auto input = InPipe::read();

				// out << "Input: ";
				// for (auto c : input) {
				// 	if (std::isprint(c)) {
				// 		out << c;
				// 	} else {
				// 		out << ".";
				// 	}
				// }
				// out << "\n";

				const auto [bitmaps, tokens] = compute_bitmaps(last_overflow_state, input);
				last_overflow_state = bitmaps.overflow_state;

				// out << "string:" << bitmaps.is_string << "\n"
				// 	<< "escapd:" << bitmaps.is_escaped << "\n";

				OutPipe::write({input, bitmaps, tokens});
			}
		});
	});

	return tokenizer_event;
}
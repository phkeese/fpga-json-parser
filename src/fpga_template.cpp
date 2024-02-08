#include <bitset>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

// oneAPI headers
#include "bitmaps.hpp"
#include "definitions.hpp"
#include "exception_handler.hpp"
#include "taped_json.hpp"
#include <fstream>
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Functions
std::vector<std::string_view> find_strings(sycl::queue &q, std::string input);
template <typename InPipe, typename OutPipe> void start_string_filter(sycl::queue &q, const size_t count);
/**
 * Write cache lines of data to a pipe.
 * @tparam WritePipe Pipe to write cache lines to.
 * @param input Data to write.
 * @return Number of cache lines to expect.
 */
template <class WritePipe> size_t write_input(sycl::queue &q, std::string &input);

// Pipes

// Host -> Bitmap computation.
using InputPipe = sycl::ext::intel::experimental::pipe<class InputPipeID, CacheLine, PIPELINE_DEPTH>;

// Host -> String filter.
using CachelineToStringFilterPipe =
	sycl::ext::intel::experimental::pipe<class CachelineToStringFilterPipeID, CacheLine, PIPELINE_DEPTH>;

// Bitmap computation -> String filter (both device).
using TokenizedCachelinesToStringFilterPipe =
	sycl::ext::intel::pipe<class TokenizedCachelinesToStringFilterPipeID, TokenizedCacheLine, PIPELINE_DEPTH>;

using OutputCacheLinePipe =
	sycl::ext::intel::experimental::pipe<class OutputCacheLinePipeID, OutputCacheLine, PIPELINE_DEPTH>;

// Main function
int main(int argc, char **argv) {
	try {
		// Use compile-time macros to select either:
		//  - the FPGA emulator device (CPU emulation of the FPGA)
		//  - the FPGA device (a real FPGA)
		//  - the simulator device
#if FPGA_SIMULATOR
		auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
		auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
		auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

		sycl::queue q(selector, fpga_tools::exception_handler, sycl::property::queue::enable_profiling{});

		auto device = q.get_device();

		std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;

		auto input = std::string{
			R"({"k":"value", "k\"y": "\"", "key": "unescaped\"", "thisisareallylongstringitinvolvesmultiplecachelines": "blub\nmore"})"};
		if (argc > 1) {
			const auto filename = argv[1];
			std::ifstream file(filename);
			if (!file.is_open()) {
				std::cerr << "Could not open file: " << filename << std::endl;
				return EXIT_FAILURE;
			}
			input = std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
		}

		auto strings = find_strings(q, input);

	} catch (sycl::exception const &e) {
		// Catches exceptions in the host code.
		std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

		// Most likely the runtime couldn't find FPGA hardware!
		if (e.code().value() == CL_DEVICE_NOT_FOUND) {
			std::cerr << "If you are targeting an FPGA, please ensure that your "
						 "system has a correctly configured FPGA board.\n";
			std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
			std::cerr << "If you are targeting the FPGA emulator, compile with "
						 "-DFPGA_EMULATOR.\n";
		}
		std::terminate();
	}

	return EXIT_SUCCESS;
}

using DebugBitmapsPipe = sycl::ext::intel::pipe<class DebugBitmapsPipeId, Bitmaps, PIPELINE_DEPTH>;

// Function definitions
std::vector<std::string_view> find_strings(sycl::queue &q, std::string input) {
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
			strings.back().append(chars.begin(), chars.begin() + string_length);
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
	std::cout << "\n taped_json.print_strings():" << std::endl;
	taped_json.print_strings();
	std::cout << "\n taped_json.print_tape():" << std::endl;
	taped_json.print_tape();

	return std::vector<std::string_view>();
}

template <class WritePipe> size_t write_input(sycl::queue &q, std::string &input) {
	// Check if input size is divisible by CACHE_LINE_SIZE
	if (input.size() % CACHE_LINE_SIZE != 0) {
		// Calculate the number of whitespaces to add
		const auto whitespace_count = CACHE_LINE_SIZE - (input.size() % CACHE_LINE_SIZE);
		// Append whitespaces to the input
		input.append(whitespace_count, ' ');
	}
	const auto line_count = input.size() / CACHE_LINE_SIZE;

	auto input_buffer = sycl::buffer{input};
	q.submit([&](auto &h) {
		auto input_accessor = sycl::accessor{input_buffer, h, sycl::read_only};
		h.template single_task([=]() {
			for (auto line_index = size_t{0}; line_index < line_count; ++line_index) {
				auto begin = input_accessor.begin() + line_index * CACHE_LINE_SIZE;
				auto end = input_accessor.begin() + (line_index + 1) * CACHE_LINE_SIZE;
				auto line = CacheLine{};
				std::copy(begin, end, line.begin());
				WritePipe::write(line);
			}
		});
	});
	return line_count;
}

template <typename InPipe, typename OutPipe> void start_string_filter(sycl::queue &q, const size_t count) {
	q.submit([&](auto &h) {
		auto out = sycl::stream(4096, 1024, h);
		h.template single_task<class StringFilterKernel>([=]() {
			for (auto index = size_t{0}; index < count; ++index) {
				const auto tokenized_cacheline = InPipe::read();
				const auto &line = tokenized_cacheline.line;
				const auto &bitmaps = tokenized_cacheline.bitmaps;

				auto current_cacheline = CacheLine{};
				auto current_count = uint16_t{0};

				auto string_lengths = CacheLine{};

				// We use a cacheline to store the lengths of the strings in the input cacheline. The first element is
				// the total number of strings, the last element indicates if there is a string overflow.
				string_lengths[0] = 0;

				for (auto byte_index = size_t{0}; byte_index < CACHE_LINE_SIZE; ++byte_index) {
					auto current_string_length = uint16_t{0};

					for (; byte_index < CACHE_LINE_SIZE && bitmaps.is_string[byte_index]; ++byte_index) {
						const auto c = line[byte_index];

						// If the current character is escaped, append the corresponding character to the output.
						if (bitmaps.is_escaped[byte_index]) {
							if (c == '\"' || c == '\\') {
								current_cacheline[current_count++] = c;
								++current_string_length;
							} else if (c == 'n') {
								current_cacheline[current_count++] = '\n';
								++current_string_length;
							} else if (c == 't') {
								current_cacheline[current_count++] = '\t';
								++current_string_length;
							} else {
								// Error: invalid escape sequence.
							}
						} else if (c != '\"' && c != '\\') {
							// If the current character is not escaped, append it to the output.s
							current_cacheline[current_count++] = c;
							++current_string_length;
						}
					}

					if (current_string_length > 0) {
						string_lengths[0] += 1;
						string_lengths[string_lengths[0]] = current_string_length;
					}
				}

				if (bitmaps.overflow_state == OverflowState::String) {
					string_lengths[CACHE_LINE_SIZE - 1] = 1;
				} else {
					string_lengths[CACHE_LINE_SIZE - 1] = 0;
				}
				if (bitmaps.is_string[0]) {
					string_lengths[CACHE_LINE_SIZE - 2] = 1;
				} else {
					string_lengths[CACHE_LINE_SIZE - 2] = 0;
				}
				/*
								for (auto token : tokenized_cacheline.tokens) {
									switch (token) {
								case Token::ObjectBeginToken:
									out << "ObjectBeginToken";
									break;
								case Token::ObjectEndToken:
									out << "ObjectEndToken";
									break;
								case Token::ArrayBeginToken:
									out << "ArrayBeginToken";
									break;
								case Token::ArrayEndToken:
									out << "ArrayEndToken";
									break;
								case Token::StringToken:
									out << "StringToken";
									break;
								case Token::FloatToken:
									out << "FloatToken";
									break;
								case Token::IntegerToken:
									out << "IntegerToken";
									break;
								case Token::EndOfTokens:
									out << "EndOfTokens";
									break;
								case Token::Testi:
									out << "Testi";
									break;
								default:
									out << "unknown";
									break;
								}
								}
								out << "\n";*/

				// Write the current cacheline to the output pipe.
				OutPipe::write({current_cacheline, string_lengths, tokenized_cacheline.tokens});
			}
		});
	});
}

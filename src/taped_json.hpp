#pragma once

#include <iostream>
#include <regex>
#include <stdexcept>
#include <vector>

#include "definitions.hpp"

union JsonValue {
	static_assert(sizeof(int64_t) == sizeof(double), "int64_t and double must be the same size");

	struct {
		size_t end_index;
		size_t saturation;
	} object_begin;
	size_t object_index;
	size_t string_index;
	int64_t integer;
	double floating_point;
};

class TapedJson {
  public:
	TapedJson() = delete;
	TapedJson(std::vector<Token> &&tokens, std::vector<std::string> &&strings)
		: _strings(std::move(strings)) {
		_construct_tape(std::move(tokens));
	}

	void print_strings() const {
		for (const auto &s : _strings) {
			std::cout << "+++" << s << "+++" << std::endl;
		}
	}

	void print_tape() const {
		for (size_t i = 0; i < _tape.size(); ++i) {
			std::cout << i << " : ";
			_print_token(std::cout, _tape[i]);
			std::cout << std::endl;
		}
	}

	// private:
	void _construct_tape(std::vector<Token> &&tokens) {
		_tape.reserve(tokens.size() + 1);
		_tape.push_back({Token::StartOfTokens, {.object_index = 0}});

		auto string_index = size_t{0};

		auto object_begin_indices = std::vector<size_t>{};
		auto object_sizes = std::vector<size_t>{0};

		for (const auto token : tokens) {
			object_sizes.back() += 1;
			switch (token) {
			case Token::ObjectBeginToken: {
				object_begin_indices.push_back(_tape.size());
				object_sizes.push_back(0);
				_tape.push_back({token, {.object_begin = {.end_index = 123123, .saturation = 456456}}});
				break;
			}
			case Token::ObjectEndToken: {
				object_sizes.back() -= 1;
				const auto object_begin_index = object_begin_indices.back();
				const auto object_size = object_sizes.back() / 2;
				// Push the object end token.
				_tape.push_back({token, {.object_index = object_begin_index}});
				// Update the object begin token with the end index and saturation.
				_tape[object_begin_index].second.object_begin = {.end_index = _tape.size(), .saturation = object_size};
				object_begin_indices.pop_back();
				object_sizes.pop_back();
				break;
			}
			case Token::ArrayBeginToken: {
				object_begin_indices.push_back(_tape.size());
				object_sizes.push_back(0);
				_tape.push_back({token, {.object_begin = {.end_index = 123123, .saturation = 456456}}});
				break;
			}
			case Token::ArrayEndToken: {
				object_sizes.back() -= 1;
				const auto object_begin_index = object_begin_indices.back();
				const auto object_size = object_sizes.back();
				// Push the object end token.
				_tape.push_back({token, {.object_index = object_begin_index}});
				// Update the object begin token with the end index and saturation.
				_tape[object_begin_index].second.object_begin = {.end_index = _tape.size(), .saturation = object_size};
				object_begin_indices.pop_back();
				object_sizes.pop_back();
				break;
			}
			case Token::StringToken:
				_tape.push_back({token, {.string_index = string_index++}});
				break;
			case Token::FloatToken:
				throw std::runtime_error("Floats are not supported");
				break;
			case Token::IntegerToken:
				throw std::runtime_error("Integers are not supported");
				break;
			case Token::EndOfTokens:
				_tape.push_back({Token::EndOfTokens, {.object_index = 0}});
				_tape[0].second.object_index = _tape.size();
				break;
			default:
				break;
			}
		}

		if (string_index != _strings.size()) {
			throw std::runtime_error("String count missmatch: " + std::to_string(string_index) + " string tokens vs " +
									 std::to_string(_strings.size()) + " strings.");
		}
	}

	void _print_token(std::ostream &os, const std::pair<Token, JsonValue> &token_value_pair) const {
		const auto &[token, value] = token_value_pair;
		switch (token) {
		case Token::StartOfTokens:
			os << "r\t// pointing to " << value.object_index << " (right after last node)";
			break;
		case Token::EndOfTokens:
			break;
		case Token::ObjectBeginToken:
			os << "{\t// pointing to next tape location " << value.object_begin.end_index
			   << " (first node after the scope),  saturated count " << value.object_begin.saturation;
			break;
		case Token::ObjectEndToken:
			os << "}\t// pointing to previous tape location " << value.object_index << " (start of the scope)";
			break;
		case Token::ArrayBeginToken:
			os << "[\t// pointing to next tape location " << value.object_begin.end_index
			   << " (first node after the scope)"
			   << ",  saturated count " << value.object_begin.saturation;
			break;
		case Token::ArrayEndToken:
			os << "]\t// pointing to previous tape location " << value.object_index << " (start of the scope)";
			break;
		case Token::StringToken: {
			auto out_string = std::regex_replace(_strings[value.string_index], std::regex(R"(\\)"), R"(\\)");
			out_string = std::regex_replace(out_string, std::regex("\""), "\\\"");
			out_string = std::regex_replace(out_string, std::regex("\n"), "\\n");
			os << "string \"" << out_string << "\"";
			break;
		}
		default:
			break;
		}
	}

	uint64_t string_count() const { return _strings.size(); }

	uint32_t max_depth() const {
		auto max_depth = uint32_t{0};
		auto current_depth = uint32_t{0};
		for (const auto &[token, value] : _tape) {
			switch (token) {
			case Token::ObjectBeginToken:
			case Token::ArrayBeginToken:
				++current_depth;
				break;
			case Token::ObjectEndToken:
			case Token::ArrayEndToken:
				--current_depth;
				break;
			default:
				break;
			}
			max_depth = std::max(max_depth, current_depth);
		}
		return max_depth;
	}

  private:
	std::vector<std::pair<Token, JsonValue>> _tape;
	std::vector<std::string> _strings;
};
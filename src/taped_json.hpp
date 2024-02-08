#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

#include "definitions.hpp"

union JsonValue {
	static_assert(sizeof(int64_t) == sizeof(double), "int64_t and double must be the same size");

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

	// void print_tokes() const {
	// 	for (const auto token : _tape) {
	// 		switch (token) {
	// 		case Token::ObjectBeginToken:
	// 			std::cout << "ObjectBeginToken" << std::endl;
	// 			break;
	// 		case Token::ObjectEndToken:
	// 			std::cout << "ObjectEndToken" << std::endl;
	// 			break;
	// 		case Token::ArrayBeginToken:
	// 			std::cout << "ArrayBeginToken" << std::endl;
	// 			break;
	// 		case Token::ArrayEndToken:
	// 			std::cout << "ArrayEndToken" << std::endl;
	// 			break;
	// 		case Token::StringToken:
	// 			std::cout << "StringToken" << std::endl;
	// 			break;
	// 		case Token::FloatToken:
	// 			std::cout << "FloatToken" << std::endl;
	// 			break;
	// 		case Token::IntegerToken:
	// 			std::cout << "IntegerToken" << std::endl;
	// 			break;
	// 		default:
	// 			std::cout << "unknown" << std::endl;
	// 			break;
	// 		}
	// 	}
	// }

	void print_strings() const {
		for (const auto &s : _strings) {
			std::cout << s << std::endl;
		}
	}

	void print_tape() const {
		std::cout << "<Begin Tape>" << std::endl;
		for (size_t i = 0; i < _tape.size(); ++i) {
			std::cout << i << " : ";
			_print_token(std::cout, _tape[i]);
			std::cout << std::endl;
		}
		std::cout << "<End Tape>" << std::endl;
	}

	// private:
	void _construct_tape(std::vector<Token> &&tokens) {
		_tape.reserve(tokens.size());
		auto string_index = size_t{0};

		auto object_stack = std::vector<size_t>{};
		for (const auto token : tokens) {
			switch (token) {
			case Token::ObjectBeginToken: {
				object_stack.push_back(_tape.size());
				_tape.push_back({token, {.object_index = 0}});
				break;
			}
			case Token::ObjectEndToken: {
				const auto object_begin_index = object_stack.back();
				object_stack.pop_back();
				_tape.push_back({token, {.object_index = object_begin_index}});
				_tape[object_begin_index].second.object_index = _tape.size();
				break;
			}
			case Token::ArrayBeginToken: {
				object_stack.push_back(_tape.size());
				_tape.push_back({token, {.object_index = 0}});
				break;
			}
			case Token::ArrayEndToken: {
				const auto object_begin_index = object_stack.back();
				object_stack.pop_back();
				_tape.push_back({token, {.object_index = object_begin_index}});
				_tape[object_begin_index].second.object_index = _tape.size();
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
			default:
				break;
			}
		}
	}

	void _print_token(std::ostream &os, const std::pair<Token, JsonValue> &token_value_pair) const {
		const auto &[token, value] = token_value_pair;
		switch (token) {
		case Token::ObjectBeginToken:
			os << "{\t// pointing to next tape location " << value.object_index << " (first node after the scope)";
			break;
		case Token::ObjectEndToken:
			os << "}\t// pointing to previous tape location " << value.object_index << " (start of the scope)";
			break;
		case Token::ArrayBeginToken:
			os << "[\t// pointing to next tape location " << value.object_index << " (first node after the scope)";
			break;
		case Token::ArrayEndToken:
			os << "]\t// pointing to previous tape location " << value.object_index << " (start of the scope)";
			break;
		case Token::StringToken:
			os << "string \"" << _strings[value.string_index] << "\"";
			break;
		default:
			break;
		}
	}

	std::vector<std::pair<Token, JsonValue>> _tape;
	std::vector<std::string> _strings;
};
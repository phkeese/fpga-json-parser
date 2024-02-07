#pragma once

#include <vector>

#include "definitions.hpp"

class TapedJson {
public:
    TapedJson() = delete;
    TapedJson(std::vector<Token>&& tape, std::vector<std::string>&& strings)
        : _tape(std::move(tape)), _strings(std::move(strings)) {}

    void print_tokes() const {
        for (const auto token : _tape) {
            switch (token) {
            case Token::ObjectBeginToken:
                std::cout << "ObjectBeginToken" << std::endl;
                break;
            case Token::ObjectEndToken:
                std::cout << "ObjectEndToken" << std::endl;
                break;
            case Token::ArrayBeginToken:
                std::cout << "ArrayBeginToken" << std::endl;
                break;
            case Token::ArrayEndToken:
                std::cout << "ArrayEndToken" << std::endl;
                break;
            case Token::StringToken:
                std::cout << "StringToken" << std::endl;
                break;
            case Token::FloatToken:
                std::cout << "FloatToken" << std::endl;
                break;
            case Token::IntegerToken:
                std::cout << "IntegerToken" << std::endl;
                break;
            default:
                std::cout << "unknown" << std::endl;
                break;
            }
        }
    }

    void print_strings() const {
        for (const auto& s : _strings) {
            std::cout << s << std::endl;
        }
    }

    void print_json() const {
        auto json_depth = size_t{0};
        auto string_count = size_t{0};

        auto many_tabs = [](const size_t count) {
            for (auto i = size_t{0}; i < count; ++i) {
                std::cout << "\t";
            }
        };

        for (const auto token : _tape) {
            switch (token) {
            case Token::ObjectBeginToken:
                many_tabs(json_depth);
                std::cout << "{";
                ++json_depth;
                break;
            case Token::ObjectEndToken:
                --json_depth;
                many_tabs(json_depth);
                std::cout << "}";
                break;
            case Token::ArrayBeginToken:
                many_tabs(json_depth);
                std::cout << "[";
                ++json_depth;
                break;
            case Token::ArrayEndToken:
                --json_depth;
                many_tabs(json_depth);
                std::cout << "]";
                break;
            case Token::StringToken:
                many_tabs(json_depth);
                std::cout << "\"" << _strings.at(string_count++) << "\"";
                break;
            case Token::FloatToken:
                many_tabs(json_depth);
                std::cout << "1.0";
                break;
            case Token::IntegerToken:
                many_tabs(json_depth);
                std::cout << "1";
                break;
            default:
                std::cout << "unknown";
                break;
            }
        }
    }


//private:
    std::vector<Token> _tape;
    std::vector<std::string> _strings;

};

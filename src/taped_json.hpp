#pragma once

#include <vector>

#include "definitions.hpp"

class TapedJson {
public:
    TapedJson() = delete;
    TapedJson(std::vector<Token>&& tape, std::vector<std::string>&& strings)
        : _tape(std::move(tape)), _strings(std::move(strings)) {}

    void print_json() const {}


//private:
    std::vector<Token> _tape;
    std::vector<std::string> _strings;

};

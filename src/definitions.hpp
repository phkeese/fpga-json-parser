#pragma once

#include <array>
#include <bitset>

// Constants
constexpr auto CACHE_LINE_SIZE = size_t{8};
constexpr auto PIPELINE_DEPTH = size_t{1};

// Types
using CacheLine = std::array<char, CACHE_LINE_SIZE>;
struct Bitmaps;

using Bitmap = std::bitset<CACHE_LINE_SIZE>;

/// Possible overflow types.
enum OverflowState {
	/// No overflow.
	None = 0,
	/// An unfinished string at the end of this line.
	String,
	/// An unfinished string with a backslash at the end.
	StringWithBackslash,
	/// Number of items in this enum.
	COUNT,
};

template <typename OS> constexpr OS &print(OS &os, OverflowState state) {
	switch (state) {
	case OverflowState::None:
		os << "None";
		break;
	case OverflowState::String:
		os << "String";
		break;
	case OverflowState::StringWithBackslash:
		os << "StringWithBackslash";
		break;
	default:
		os << "unknown";
		break;
	}
	return os;
}

struct Overflows {
	bool string_overflow = false;
	bool backslash_overflow = false;
	bool number_overflow = false;
};

struct Bitmaps {
	CacheLine input;
	Bitmap is_string;
	Bitmap is_escaped;
	OverflowState overflow_state;
	Overflows overflows;
};
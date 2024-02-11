#pragma once

#include <array>
#include <bitset>

// Constants
constexpr auto CACHE_LINE_SIZE = size_t{64};
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

enum Token : uint8_t {
	EndOfTokens = 0,
	StartOfTokens = 1,
	ObjectBeginToken,
	ObjectEndToken,
	ArrayBeginToken,
	ArrayEndToken,
	StringToken,
	FloatToken,
	IntegerToken
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

struct Bitmaps {
	// CacheLine input;
	Bitmap is_string;
	Bitmap is_escaped;
	OverflowState overflow_state;
};

struct TokenizedCacheLine {
	CacheLine line;
	Bitmaps bitmaps;
	CacheLine tokens;
};

struct OutputCacheLine {
	CacheLine line;
	CacheLine string_lengths;
	CacheLine tokens;
};

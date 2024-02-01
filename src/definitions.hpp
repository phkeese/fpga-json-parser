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
enum OverflowState : uint8_t {
	/// No overflow.
	None = 0,
	/// An unfinished string at the end of this line.
	String,
	/// An unfinished string with a backslash at the end.
	StringWithBackslash,
	/// Number of items in this enum.
	COUNT,
};

constexpr auto OverflowStateStrings = std::array{
	"None",
	"String",
	"StringWithBackslash",
};

struct Overflows {
	bool string_overflow = false;
	bool backslash_overflow = false;
	bool number_overflow = false;
};

struct Bitmaps {
	Bitmap is_string;
	Bitmap is_escaped;
	OverflowState overflow_state;
	Overflows overflows;
};
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

struct TokenizedCacheline {
	CacheLine line;
	Bitmaps bitmaps;
};

struct OutputCacheLine {
	CacheLine line;
	CacheLine string_lengths;
};

// //
// // Extend a type 'T' with a boolean flag
// //
// template <typename T>
// struct FlagBundle {
//   using value_type = T;

//   // ensure the type carried in this class has a subscript operator and that
//   // it has a static integer member named 'size'
//   static_assert(fpga_tools::has_subscript_v<T>);

//   // this is used by the functions in memory_utils.hpp to ensure the size of
//   // the type in the SYCL pipe matches the memory width
//   static constexpr size_t size = T::size;

//   FlagBundle() : data(T()), flag(false) {}
//   FlagBundle(T d_in) : data(d_in), flag(false) {}
//   FlagBundle(T d_in, bool f_in) : data(d_in), flag(f_in) {}
//   FlagBundle(bool f_in) : data(T()), flag(f_in) {}

//   unsigned char& operator[](int i) { return data[i]; }
//   const unsigned char& operator[](int i) const { return data[i]; }

//   T data;
//   bool flag;
// };


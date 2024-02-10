#pragma once

template <typename Id, typename InPipe, typename OutPipe>
sycl::event submit_string_filter(sycl::queue &q, const size_t count) {
	const auto string_filter_event = q.submit([&](auto &h) {
		h.template single_task<Id>([=]() {
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
					auto started_string = bool{false};
					auto current_string_length = uint16_t{0};

					for (; byte_index < CACHE_LINE_SIZE && bitmaps.is_string[byte_index]; ++byte_index) {
						started_string = true;
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

					if (started_string) {
						string_lengths[0] += 1;
						string_lengths[string_lengths[0]] = current_string_length;
					}
				}

				if (bitmaps.is_string[CACHE_LINE_SIZE - 1]) {
					string_lengths[CACHE_LINE_SIZE - 1] = 1;
				} else {
					string_lengths[CACHE_LINE_SIZE - 1] = 0;
				}
				if (bitmaps.is_string[0]) {
					string_lengths[CACHE_LINE_SIZE - 2] = 1;
				} else {
					string_lengths[CACHE_LINE_SIZE - 2] = 0;
				}

				// Write the current cacheline to the output pipe.
				OutPipe::write({current_cacheline, string_lengths, tokenized_cacheline.tokens});
			}
		});
	});

	return string_filter_event;
}

#pragma once

#include <iostream>
#include <sstream>

struct LogStream {
	bool to_err_;
	std::ostringstream buf_;

	LogStream(bool to_err, const char *level, const char *tag) : to_err_(to_err) {
		buf_ << "[" << level << "][" << tag << "] ";
	}

	~LogStream() {
		(to_err_ ? std::cerr : std::cout) << buf_.str() << "\n";
	}

	template <typename T>
	LogStream &operator<<(const T &v) {
		buf_ << v;
		return *this;
	}
};

#define LOG_INFO(tag) LogStream(false, "INFO", tag)
#define LOG_WARN(tag) LogStream(false, "WARN", tag)
#define LOG_ERROR(tag) LogStream(true, "ERROR", tag)

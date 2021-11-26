/**
 * \file utils.hpp
 * \author smh
 * \date 2020.04.23
 * \copyright Copyright (c) 2020 sogou-inc
 */

#ifndef UTILS_HPP__
#define UTILS_HPP__

#include <string>

#include <sys/stat.h>
#include <sys/types.h>

bool is_file_exists(const std::string& in_file);

#endif

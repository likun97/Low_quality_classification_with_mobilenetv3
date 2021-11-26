/**
 * \file utils.cpp
 * \author smh
 * \date 2020.04.23
 * \copyright Copyright (c) 2020 sogou-inc
 */
#include "utils.hpp"

bool is_file_exists(const std::string& in_file)
{
    struct stat buf;
    if (stat(in_file.c_str(), &buf) == 0)
        return true;
    else
        return false;
}

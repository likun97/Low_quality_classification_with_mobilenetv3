/*************************************************************************
    > File Name: test.cpp
    > Author:  
    > Created Time: Thu 09 Mar 2017 10:31:13 AM CST
 ************************************************************************/
//#ifndef SYS_HPP
//#define SYS_HPP

//#include <string>
//#include <vector>
#include "sys.hpp"
namespace daka
{

    void split(std::string& src, std::vector<std::string>& dest, std::string& delim)
{
    size_t last = 0;
    size_t index=src.find_first_of(delim,last);
     while (index!=std::string::npos)
     {
         dest.push_back(src.substr(last,index-last));
         last=index+1;
         index=src.find_first_of(delim,last);
     }
     if (index-last>0)
     {
         dest.push_back(src.substr(last,index-last)); 
     }
}

};

//#endif

#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "utils.h"

using namespace LAMMPS_NS;

double utils::numeric(const char*, int, std::string line, bool, char*)
{
    return atof(line.c_str());
}

double utils::inumeric(const char*, int, std::string line, bool, char*)
{
    return atoi(line.c_str());
}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn trim_comment
///
/// \brief <insert brief description>
///
/// <Insert longer more detailed description which
/// can span multiple lines if needed>
///
/// \param <function parameter description>
///
/// \return <return type and definition description if not void>
///
/////////////////////////////////////////////////////////////////////////////
std::string utils::trim_comment(const std::string& line)
{
    auto end = line.find("#");
    if (end != std::string::npos) {
        return line.substr(0, end);
    }
    return { line };
}

FILE* utils::open_potential(char* filename, char*, std::nullptr_t)
{
    return fopen(filename, "r");
}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn strdup
///
/// \brief <insert brief description>
///
/// <Insert longer more detailed description which
/// can span multiple lines if needed>
///
/// \param <function parameter description>
///
/// \return <return type and definition description if not void>
///
/////////////////////////////////////////////////////////////////////////////
char* utils::strdup(const std::string& text)
{
    auto tmp = new char[text.size() + 1];
    strcpy(tmp, text.c_str());
    return tmp;
}
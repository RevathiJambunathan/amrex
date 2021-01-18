#ifndef PARSER_UTILS_H
#define PARSER_UTILS_H

#include <string>
#include <AMReX_ParmParse.H>
#include "WarpXParser.H"

void Store_parserString( const amrex::ParmParse &pp, std::string query_string,
                         std::string& stored_string);

WarpXParser makeParser (std::string const& parse_function, std::vector<std::string> const& varnames);

#endif

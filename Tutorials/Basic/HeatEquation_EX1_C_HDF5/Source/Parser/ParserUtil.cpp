#include "ParserUtil.H"
#include <AMReX_ParmParse.H>
#include <cmath>
#include <fstream>

void Store_parserString (const amrex::ParmParse& pp, std::string query_string,
                         std::string& stored_string)
{
    std::vector<std::string> f;
    pp.getarr(query_string.c_str(), f);
    stored_string.clear();
    for (auto const& s : f) {
        stored_string += s;
    }
    f.clear();
}


WarpXParser makeParser (std::string const& parse_function, std::vector<std::string> const& varnames)
{
    WarpXParser parser(parse_function);
    parser.registerVariables(varnames);
    amrex::ParmParse pp("my_constants");
    std::set<std::string> symbols = parser.symbols();
    for (auto const& v : varnames) symbols.erase(v.c_str());
    for (auto it = symbols.begin(); it != symbols.end(); ) {
        amrex::Real v;
        if (pp.query(it->c_str(), v)) {
            parser.setConstant(*it, v);
            it = symbols.erase(it);
        } else {
            ++it;
        }
    }
    for (auto const& s : symbols) {
        amrex::Abort("makeParser::Unknown symbol "+s);
    }
    return parser;
}

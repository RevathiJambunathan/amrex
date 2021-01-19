#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include "Parser/WarpXParserWrapper.H"
#include "Parser/GpuParser.H"

std::string phi_init_type;
std::string str_phi_init_function;
std::unique_ptr<ParserWrapper<3> > phi_parser;
#endif

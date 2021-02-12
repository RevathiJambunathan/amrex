#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include "Parser/ParserWrapper.H"
#include "Parser/GpuParser.H"

std::string phi_init_type;
std::string str_phi_init_function;
std::unique_ptr<ParserWrapper<3> > phi_parser;
amrex::Real diffusion_coefficient;
#endif

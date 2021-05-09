#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include "Parser/ParserWrapper.H"
#include "Parser/GpuParser.H"

std::string phi_init_type;
std::string str_phi_init_function;
std::unique_ptr<ParserWrapper<3> > phi_parser;
amrex::Real diffusion_coefficient;
int num_gaussians;
amrex::Gpu::DeviceVector<amrex::Real> amplitude_min;
amrex::Gpu::DeviceVector<amrex::Real> amplitude_max;
amrex::Gpu::DeviceVector<amrex::Real> sigmax_min;
amrex::Gpu::DeviceVector<amrex::Real> sigmax_max;
amrex::Gpu::DeviceVector<amrex::Real> sigmay_min;
amrex::Gpu::DeviceVector<amrex::Real> sigmay_max;
amrex::Gpu::DeviceVector<amrex::Real> xc;
amrex::Gpu::DeviceVector<amrex::Real> yc;
amrex::Gpu::DeviceVector<amrex::Real> sigmax;
amrex::Gpu::DeviceVector<amrex::Real> sigmay;
amrex::Gpu::DeviceVector<amrex::Real> theta;
amrex::Gpu::DeviceVector<amrex::Real> amplitude;

#endif

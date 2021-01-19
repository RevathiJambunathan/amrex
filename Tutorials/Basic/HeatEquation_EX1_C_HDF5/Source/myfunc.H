#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <iostream>
#include "Parser/ParserWrapper.H"
#include "Parser/GpuParser.H"

void main_main ();

void advance (amrex::MultiFab& phi_old,
              amrex::MultiFab& phi_new,
	      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
	      amrex::Real dt,
              amrex::Geometry const& geom);

void init_phi (amrex::MultiFab& phi_new, amrex::Geometry const& geom);


void init_phi_withparser( amrex::MultiFab& phi_new, amrex::Geometry const& geom,
                          HostDeviceParser<3> const& phi_parser);
#endif

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
              amrex::Geometry const& geom, amrex::Real diffusion_coefficient);

void init_phi (amrex::MultiFab& phi_new, amrex::Geometry const& geom);


void init_phi_withparser( amrex::MultiFab& phi_new, amrex::Geometry const& geom,
                          HostDeviceParser<3> const& phi_parser);

#if (AMREX_SPACEDIM==2)
void init_phi_generic_2Dgaussian_RandomParameters(amrex::MultiFab& phi_new,
         amrex::Geometry const& geom,
         amrex::Real * AMREX_RESTRICT const amplitude_min,
         amrex::Real * AMREX_RESTRICT const amplitude_max,
         amrex::Real * AMREX_RESTRICT const sigmax_min,
         amrex::Real * AMREX_RESTRICT const sigmax_max,
         amrex::Real * AMREX_RESTRICT const sigmay_min,
         amrex::Real * AMREX_RESTRICT const sigmay_max,
         amrex::Real * AMREX_RESTRICT const sigmax,
         amrex::Real * AMREX_RESTRICT const sigmay,
         amrex::Real * AMREX_RESTRICT const amplitude,
         amrex::Real * AMREX_RESTRICT const theta,
         amrex::Real * AMREX_RESTRICT const xc,
         amrex::Real * AMREX_RESTRICT const yc,
         int const num_gaussians);
#endif

#endif

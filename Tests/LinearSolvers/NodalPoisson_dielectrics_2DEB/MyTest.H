#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MLMG.H>
#include <AMReX_EBFabFactory.H>

// Solve div(epsilon(x,y) grad phi) = 0
// epsilon(r) = eps1  for r = sqrt(x^2+y^2) < a   (inside cylinder)
//              eps2  for r >= a                    (outside cylinder)
// Domain [-L, L]^2; Dirichlet BCs at x = ±L (uniform applied field),
// Neumann BCs at y = ±L.
// Compiled with AMREX_USE_EB; an all-regular EBFArrayBoxFactory is passed
// to MLEBNodeFDLaplacian so the solver uses its non-EB kernel path while
// the dielectric variation is handled entirely through the cell-centered sigma.

class MyTest
{
public:

    MyTest ();

    void solve ();
    void compute_norms () const;
    void writePlotfile () const;

    void initData ();

    [[nodiscard]] int getNumTrials () const { return 1; }
    [[nodiscard]] int getDoPlots () const { return do_plots; }

private:

    void readParameters ();

    int n_cell = 128;
    int max_grid_size = 64;
    int max_level = 0;
    int ref_ratio = 2;

    // Problem geometry and material parameters
    amrex::Real L_domain = 3.0;   // half-width: domain is [-L, L]^2
    amrex::Real a_cyl    = 1.0;   // cylinder radius
    amrex::Real eps1     = 5.0;   // permittivity inside cylinder
    amrex::Real eps2     = 1.0;   // permittivity outside cylinder
    amrex::Real E0       = 1.0;   // applied far-field magnitude

    // MLMG solver settings
    int verbose = 2;
    int bottom_verbose = 0;
    int max_iter = 200;
    int max_fmg_iter = 0;
    amrex::Real reltol = 1.e-10;

    int gpu_regtest = 0;
    bool do_plots = true;

    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;

    std::unique_ptr<amrex::EBFArrayBoxFactory> factory;

    amrex::Vector<amrex::MultiFab> solution;
    amrex::Vector<amrex::MultiFab> rhs;
    amrex::Vector<amrex::MultiFab> exact_solution;
    amrex::Vector<amrex::MultiFab> sigma;
    amrex::Vector<amrex::MultiFab> electric_field;  // cell-centered, AMREX_SPACEDIM components
};

#endif

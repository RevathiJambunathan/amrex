#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include "Parser/ParserWrapper.H"
#include "Parser/ParserUtil.H"

#include "main.H"
#include "myfunc.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    main_main();
    
    amrex::Finalize();
    return 0;
}

void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    Real strt_time = ParallelDescriptor::second();

    // AMREX_SPACEDIM: number of dimensions
    int n_cell, max_grid_size, nsteps, plot_int;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of 
        //   a square (or cubic) domain.
        pp.get("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        phi_init_type = "default";
        pp.query("phi_init_type", phi_init_type);
        amrex::Print() << " phi_init_type " << phi_init_type << "\n";
        if ( phi_init_type == "parse_phi_function" ) {
            Store_parserString(pp, "phi_init_function(x,y,z)", str_phi_init_function);
            phi_parser = std::make_unique<ParserWrapper<3>>(
                             makeParser(str_phi_init_function,{"x","y","z"}));
        }
 
        // Default diffusion_coefficient = 1
        diffusion_coefficient = 1.;
        pp.query("diffusionCoefficient", diffusion_coefficient);
      
        
#if (AMREX_SPACEDIM==2)
        pp.get("num_gaussians", num_gaussians);
        amrex::Vector<amrex::Real> amplitude_min_v(num_gaussians); 
        amrex::Vector<amrex::Real> amplitude_max_v(num_gaussians); 
        amrex::Vector<amrex::Real> sigmax_min_v(num_gaussians); 
        amrex::Vector<amrex::Real> sigmax_max_v(num_gaussians); 
        amrex::Vector<amrex::Real> sigmay_min_v(num_gaussians); 
        amrex::Vector<amrex::Real> sigmay_max_v(num_gaussians); 
        amrex::Vector<amrex::Real> xc_v(num_gaussians); 
        amrex::Vector<amrex::Real> yc_v(num_gaussians); 

        amplitude_min.resize(num_gaussians);
        amplitude_max.resize(num_gaussians);
        sigmax_min.resize(num_gaussians);
        sigmax_max.resize(num_gaussians);
        sigmay_min.resize(num_gaussians);
        sigmay_max.resize(num_gaussians);
        xc.resize(num_gaussians);
        yc.resize(num_gaussians);
        theta.resize(num_gaussians);
        amplitude.resize(num_gaussians);
        sigmax.resize(num_gaussians);
        sigmay.resize(num_gaussians);

        pp.getarr("amplitude_min", amplitude_min_v, 0, num_gaussians);
        pp.getarr("amplitude_max", amplitude_max_v, 0, num_gaussians);
        pp.getarr("sigmax_min", sigmax_min_v, 0, num_gaussians);
        pp.getarr("sigmay_min", sigmay_min_v, 0, num_gaussians);
        pp.getarr("sigmax_max", sigmax_max_v, 0, num_gaussians);
        pp.getarr("sigmay_max", sigmay_max_v, 0, num_gaussians);
        std::copy(amplitude_min_v.begin(), amplitude_min_v.end(), amplitude_min.begin());
        std::copy(amplitude_max_v.begin(), amplitude_max_v.end(), amplitude_max.begin());
        std::copy(sigmax_min_v.begin(), sigmax_min_v.end(), sigmax_min.begin());
        std::copy(sigmax_max_v.begin(), sigmax_max_v.end(), sigmax_max.begin());
        std::copy(sigmay_min_v.begin(), sigmay_min_v.end(), sigmay_min.begin());
        std::copy(sigmay_max_v.begin(), sigmay_max_v.end(), sigmay_max.begin());

#endif

    }

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

       // This defines the physical box, [-1,1] in each direction.
        RealBox real_box({AMREX_D_DECL(-Real(1.0),-Real(1.0),-Real(1.0))},
                         {AMREX_D_DECL( Real(1.0), Real(1.0), Real(1.0))});

        // periodic in all direction
        Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

        // This defines a Geometry object
        geom.define(domain,real_box,CoordSys::cartesian,is_periodic);
    }

    // Nghost = number of ghost cells for each array 
    int Nghost = 1;
    
    // Ncomp = number of components for each array
    int Ncomp  = 1;
  
    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab phi_old(ba, dm, Ncomp, Nghost);
    MultiFab phi_new(ba, dm, Ncomp, Nghost);

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    if (phi_init_type == "default") {
        init_phi(phi_new, geom);
    } else if (phi_init_type == "parse_phi_function") {
        init_phi_withparser(phi_new, geom, getParser(phi_parser));
#if (AMREX_SPACEDIM==2)    
    } else if (phi_init_type == "generic_2Dgaussian_RandomParameters") {
        init_phi_generic_2Dgaussian_RandomParameters(phi_new, geom,
            amplitude_min.dataPtr(), amplitude_max.dataPtr(),
            sigmax_min.dataPtr(), sigmax_max.dataPtr(),
            sigmay_min.dataPtr(), sigmay_max.dataPtr(),
            sigmax.dataPtr(), sigmay.dataPtr(), amplitude.dataPtr(), theta.dataPtr(),
            xc.dataPtr(), yc.dataPtr(), num_gaussians);
#endif
    }
    // ========================================

    Real cfl = 0.9;
    Real coeff = AMREX_D_TERM(   1./(dx[0]*dx[0]),
                               + 1./(dx[1]*dx[1]),
                               + 1./(dx[2]*dx[2]) );
    Real dt = cfl/(2.0*coeff);

    // time = starting time in the simulation
    Real time = 0.0;

    // Write a plotfile of the initial data if plot_int > 0 (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        int n = 0;
        const std::string& pltfile = amrex::Concatenate("plt",n,5);
#ifdef AMREX_USE_HDF5
        WriteSingleLevelPlotfileHDF5(pltfile, phi_new, {"phi"}, geom, time, 0);
#else
        WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, 0);
#endif
    }

    // build the flux multifabs
    Array<MultiFab, AMREX_SPACEDIM> flux;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        // flux(dir) has one component, zero ghost cells, and is nodal in direction dir
        BoxArray edge_ba = ba;
        edge_ba.surroundingNodes(dir);
        flux[dir].define(edge_ba, dm, 1, 0);
    }

    for (int n = 1; n <= nsteps; ++n)
    {
        MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);

        // new_phi = old_phi + dt * (something)
        advance(phi_old, phi_new, flux, dt, geom, diffusion_coefficient);
        time = time + dt;
        
        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << n << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",n,5);
#ifdef AMREX_USE_HDF5
            WriteSingleLevelPlotfileHDF5(pltfile, phi_new, {"phi"}, geom, time, n);
#else
            WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, n);
#endif
        }
    }

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    Real stop_time = ParallelDescriptor::second() - strt_time;
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;
}

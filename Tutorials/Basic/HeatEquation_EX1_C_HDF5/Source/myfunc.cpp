
#include "myfunc.H"
#include "mykernel.H"

using namespace amrex;

void advance (MultiFab& phi_old,
              MultiFab& phi_new,
	      Array<MultiFab, AMREX_SPACEDIM>& flux,
	      Real dt,
              Geometry const& geom, amrex::Real diffusion_coefficient)
{

    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries.
    // There are no physical domain boundaries to fill in this example.
    phi_old.FillBoundary(geom.periodicity());

    //
    // Note that this simple example is not optimized.
    // The following two MFIter loops could be merged
    // and we do not have to use flux MultiFab.
    // 
    // =======================================================

    // This example supports both 2D and 3D.  Otherwise,
    // we would not need to use AMREX_D_TERM.
    AMREX_D_TERM(const Real dxinv = geom.InvCellSize(0);,
                 const Real dyinv = geom.InvCellSize(1);,
                 const Real dzinv = geom.InvCellSize(2););

    // Compute fluxes one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.nodaltilebox(0);
        const Box& ybx = mfi.nodaltilebox(1);
        auto const& fluxx = flux[0].array(mfi);
        auto const& fluxy = flux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.nodaltilebox(2);
        auto const& fluxz = flux[2].array(mfi);
#endif
        auto const& phi = phi_old.const_array(mfi);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_x(i,j,k,fluxx,phi,dxinv,diffusion_coefficient);
        });

        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_y(i,j,k,fluxy,phi,dyinv,diffusion_coefficient);
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_z(i,j,k,fluxz,phi,dzinv,diffusion_coefficient);
        });
#endif
    }

    // Advance the solution one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& fluxx = flux[0].const_array(mfi);
        auto const& fluxy = flux[1].const_array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& fluxz = flux[2].const_array(mfi);
#endif
        auto const& phiOld = phi_old.const_array(mfi);
        auto const& phiNew = phi_new.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            update_phi(i,j,k,phiOld,phiNew,
                       AMREX_D_DECL(fluxx,fluxy,fluxz),
                       dt,
                       AMREX_D_DECL(dxinv,dyinv,dzinv));
        });
    }
}

void init_phi(amrex::MultiFab& phi_new, amrex::Geometry const& geom){

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    // =======================================
    // Initialize phi_new by calling a Fortran routine.
    // MFIter = MultiFab Iterator
    for (MFIter mfi(phi_new); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto const& phiNew = phi_new.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_phi(i,j,k,phiNew,dx,prob_lo);
        });
    }
}

/** Initialize phi by calling user-defined parser function */
void init_phi_withparser( amrex::MultiFab& phi_new, amrex::Geometry const& geom,
                          HostDeviceParser<3> const& phi_parser)
{
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    amrex::IntVect mf_nodal_flag = phi_new.ixType().toIntVect();
    for (MFIter mfi(phi_new); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& phi_arr = phi_new.array(mfi);
 
        amrex::ParallelFor (vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Determine x co-ordinate corresponding to index i
            amrex::Real fac_x = (1._rt - mf_nodal_flag[0]) * dx[0] * 0.5_rt;
            amrex::Real x = i * dx[0] + prob_lo[0] + fac_x;

            // Determine y co-ordinate corresponding to index j
            amrex::Real fac_y = (1._rt - mf_nodal_flag[1]) * dx[1] * 0.5_rt;
            amrex::Real y = j * dx[1] + prob_lo[1] + fac_y;

            // Determine z co-ordinate corresponding to index k
            amrex::Real fac_z = (1._rt - mf_nodal_flag[2]) * dx[2] * 0.5_rt;
            amrex::Real z = k * dx[2] + prob_lo[2] + fac_z;

            // initialize phi at (x,y,z);
            phi_arr(i,j,k) = phi_parser(x,y,z);
        });
    }
}

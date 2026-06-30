#include "MyTest.H"

#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_EB2.H>
#include <AMReX_EBMultiFabUtil.H>

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();
    initData();
}

void
MyTest::solve ()
{
    BL_PROFILE("NodalPoissonDielectrics2DEB::solve()");

    LPInfo info;

    // EB-factory overload: compiled with AMREX_USE_EB but factory is all-regular
    // (no cut cells), so the solver runs its non-EB kernel path.
    // The dielectric variation is handled entirely through the cell-centered sigma.
    MLEBNodeFDLaplacian linop(geom, grids, dmap, info,
                              {factory.get()});

    // x-faces: Dirichlet (applied far-field).  y-faces: Neumann (homogeneous).
    linop.setDomainBC(
        {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Neumann,  LinOpBCType::Neumann)},
        {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Neumann,  LinOpBCType::Neumann)});

    linop.setSigma(0, sigma[0]);

    MLMG mlmg(linop);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);

    // Initial guess: zero everywhere.
    solution[0].setVal(0.0);

    // Dirichlet BCs at x = -L and x = +L from the uniform applied field:
    //   phi(-L, y) = +E0*L,   phi(+L, y) = -E0*L
    const Box node_domain = amrex::surroundingNodes(geom[0].Domain());
    const int ilo = node_domain.smallEnd(0);
    const int ihi = node_domain.bigEnd(0);

    Box lo_face = node_domain;
    lo_face.setRange(0, ilo, 1);
    Box hi_face = node_domain;
    hi_face.setRange(0, ihi, 1);

    solution[0].setVal( E0 * L_domain, lo_face, 0, 1);
    solution[0].setVal(-E0 * L_domain, hi_face, 0, 1);

    mlmg.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs), reltol, 0.0);

    // Compute E = -grad phi via compGrad.
    // compGrad requires:
    //   grad[0]: index type (0,1) — cell-centered in x, nodal in y
    //   grad[1]: index type (1,0) — nodal in x, cell-centered in y
    // solution must have ghost cells filled before the call.
    solution[0].FillBoundary(geom[0].periodicity());

    Array<MultiFab, AMREX_SPACEDIM> grad;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        IntVect typ = IntVect::TheNodeVector();
        typ[idim] = 0;
        BoxArray ba = amrex::convert(grids[0], typ);
        grad[idim].define(ba, dmap[0], 1, 1);
    }
    linop.compGrad(0, {AMREX_D_DECL(&grad[0], &grad[1], &grad[2])},
                   solution[0], MLLinOp::Location::FaceCenter);

    // Average edge-centered grad components to cell-center and negate (E = -grad phi).
    // grad[0] type (0,1): cell-centered in x, nodal in y → average over j
    // grad[1] type (1,0): nodal in x, cell-centered in y → average over i
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(electric_field[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Array4<Real>       const ef = electric_field[0].array(mfi);
        Array4<Real const> const gx = grad[0].const_array(mfi);
        Array4<Real const> const gy = grad[1].const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            ef(i,j,k,0) = -Real(0.5) * (gx(i,j,k) + gx(i,j+1,k));  // Ex
            ef(i,j,k,1) = -Real(0.5) * (gy(i,j,k) + gy(i+1,j,k));  // Ey
        });
    }
}

void
MyTest::compute_norms () const
{
    MultiFab error(solution[0].boxArray(), solution[0].DistributionMap(), 1, 0);
    MultiFab::Copy(error, solution[0], 0, 0, 1, 0);
    MultiFab::Subtract(error, exact_solution[0], 0, 0, 1, 0);

    auto mask = error.OwnerMask(geom[0].periodicity());

    amrex::Print() << "max-norm: " << error.norm0(*mask, 0, 0) << "\n";
    const Real* dx = geom[0].CellSize();
    Real dvol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    amrex::Print() << "1-norm  : " << error.norm1(0, geom[0].periodicity())*dvol << "\n";
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);

    pp.query("L_domain", L_domain);
    pp.query("a_cyl", a_cyl);
    pp.query("eps1", eps1);
    pp.query("eps2", eps2);
    pp.query("E0", E0);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("reltol", reltol);
#ifdef AMREX_USE_FLOAT
    reltol = std::max(reltol, 1.e-5F);
#endif

    pp.query("gpu_regtest", gpu_regtest);
    pp.query("do_plots", do_plots);
}

void
MyTest::initData ()
{
    geom.resize(1);
    grids.resize(1);
    dmap.resize(1);
    solution.resize(1);
    rhs.resize(1);
    exact_solution.resize(1);
    sigma.resize(1);
    electric_field.resize(1);

    // 2D domain: [-L, L] x [-L, L]
    RealBox rb({AMREX_D_DECL(-L_domain, -L_domain, 0.)},
               {AMREX_D_DECL( L_domain,  L_domain, 1.)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, is_periodic.data());

    Box domain(IntVect{AMREX_D_DECL(0,0,0)},
               IntVect{AMREX_D_DECL(n_cell-1, n_cell-1, 0)});
    geom[0].define(domain);

    grids[0].define(domain);
    grids[0].maxSize(max_grid_size);
    dmap[0].define(grids[0]);

    // Build an all-regular EB index space (eb2.geom_type = all_regular in inputs).
    // No cut cells — the cylinder is represented only through sigma.
    EB2::Build(geom[0], 0, 100);

    factory = makeEBFabFactory(geom[0], grids[0], dmap[0],
                               {2,2,2}, EBSupport::full);

    const BoxArray& nba = amrex::convert(grids[0], IntVect::TheNodeVector());

    // phi and rhs are nodal; sigma and electric_field are cell-centered
    solution      [0].define(nba,      dmap[0], 1,              1, MFInfo(), *factory);
    rhs           [0].define(nba,      dmap[0], 1,              0, MFInfo(), *factory);
    exact_solution[0].define(nba,      dmap[0], 1,              0, MFInfo(), *factory);
    sigma         [0].define(grids[0], dmap[0], 1,              1, MFInfo(), *factory);
    electric_field[0].define(grids[0], dmap[0], AMREX_SPACEDIM, 0, MFInfo(), *factory);

    rhs[0].setVal(0.0);   // no free charges

    const auto dx = geom[0].CellSizeArray();
    const auto problo = geom[0].ProbLoArray();

    const Real e1 = eps1, e2 = eps2, a = a_cyl, e0 = E0;
    const Real coeff = (e1 - e2) / (e1 + e2) * a * a;

    // Analytic nodal solution — stored only for post-solve error analysis,
    // not used anywhere in the numerical setup.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(exact_solution[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Array4<Real> const phi = exact_solution[0].array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = problo[0] + i * dx[0];
            Real y = problo[1] + j * dx[1];
            Real r2 = x*x + y*y;
            Real phi_val;
            if (r2 < a*a) {
                phi_val = -Real(2.0)*e2/(e1+e2) * e0 * x;
            } else {
                phi_val = -e0*x + e0 * coeff * x / r2;
            }
            phi(i,j,k) = phi_val;
        });
    }

    // Piecewise constant sigma: cell center at problo + (i+0.5)*dx
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(sigma[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();
        Array4<Real> const sig = sigma[0].array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = problo[0] + (i + Real(0.5)) * dx[0];
            Real y = problo[1] + (j + Real(0.5)) * dx[1];
            Real r2 = x*x + y*y;
            sig(i,j,k) = (r2 < a*a) ? e1 : e2;
        });
    }
}

#ifndef AMREX_RAND_H
#define AMREX_RAND_H

#include <AMReX.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RandomEngine.H>
#include <limits>
#include <cstdint>

namespace amrex
{
    /**
    * \brief Generate a psuedo-random double from uniform distribution
    *
    *  Generates one pseudorandom real number (double) from a uniform
    *  distribution between 0.0 and 1.0 (0.0 included, 1.0 excluded)
    *
    */
#ifdef AMREX_USE_CUDA
    AMREX_GPU_HOST_DEVICE Real Random ();
#else
    Real Random ();
#endif

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real Random (RandomEngine const& random_engine)
    {
#if defined(__CUDA_ARCH__)
#ifdef BL_USE_FLOAT
        return 1.0f - curand_uniform(random_engine.rand_state);
#else
        return 1.0 - curand_uniform_double(random_engine.rand_state);
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
#ifdef BL_USE_FLOAT
        return 1.0f - hiprand_uniform(random_engine.rand_state);
#else
        return 1.0 - hiprand_uniform_double(random_engine.rand_state);
#endif
#elif defined (__SYCL_DEVICE_ONLY__)
        mkl::rng::device::uniform<Real> distr;
        return mkl::rng::device::generate(distr, *random_engine.engine);
#else
        amrex::ignore_unused(random_engine);
        return Random();
#endif
    }

    /**
    * \brief Generate a psuedo-random double from a normal distribution
    *
    *  Generates one pseudorandom real number (double) from a normal
    *  distribution with mean 'mean' and standard deviation 'stddev'.
    *
    */
#ifdef AMREX_USE_CUDA
    AMREX_GPU_HOST_DEVICE Real RandomNormal (Real mean, Real stddev);
#else
    Real RandomNormal (Real mean, Real stddev);
#endif

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real RandomNormal (Real mean, Real stddev, RandomEngine const& random_engine)
    {
#if defined(__CUDA_ARCH__)
#ifdef BL_USE_FLOAT
        return stddev * curand_normal(random_engine.rand_state) + mean;
#else
        return stddev * curand_normal_double(random_engine.rand_state) + mean;
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
#ifdef BL_USE_FLOAT
        return stddev * hiprand_normal(random_engine.rand_state) + mean;
#else
        return stddev * hiprand_normal_double(random_engine.rand_state) + mean;
#endif
#elif defined (__SYCL_DEVICE_ONLY__)
        mkl::rng::device::gaussian<Real> distr(mean, stddev);
        return mkl::rng::device::generate(distr, *random_engine.engine);
#else
        amrex::ignore_unused(random_engine);
        return RandomNormal(mean, stddev);
#endif
    }

    /**
    * \brief Generate a psuedo-random integer from a Poisson distribution
    *
    *  Generates one pseudorandom positive integer number (double)
    *  extracted from a Poisson distribution, given the Real parameter lambda.
    *  The CPU version of this function relies on the standard Template Library
    *  The GPU version of this function relies on the cuRAND library
    *
    */
#ifdef AMREX_USE_CUDA
    AMREX_GPU_HOST_DEVICE unsigned int RandomPoisson (Real lambda);
#else
    unsigned int RandomPoisson (Real lambda);
#endif

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    unsigned int RandomPoisson (Real lambda, RandomEngine const& random_engine)
    {
#if defined(__CUDA_ARCH__)
        return curand_poisson(random_engine.rand_state, lambda);
#elif defined(__HIP_DEVICE_COMPILE__)
        return hiprand_poisson(random_engine.rand_state, lambda);
#elif defined (__SYCL_DEVICE_ONLY__)
        amrex::ignore_unused(lambda,random_engine);
        amrex::Abort("RandomPossion not supported in DPC++ device code");
        return 0; // xxxxx DPCPP todo: Poisson distribution
#else
        amrex::ignore_unused(random_engine);
        return RandomPoisson(lambda);
#endif
    }

    /**
    * \brief Generates one pseudorandom unsigned integer which is
    *  uniformly distributed on [0,n-1]-interval for each call.
    *
    * The CPU version of this function uses C++11's mt19937.
    * The GPU version uses CURAND's XORWOW generator.
    */
#ifdef AMREX_USE_CUDA
    AMREX_GPU_HOST_DEVICE unsigned int Random_int (unsigned int n); // [0,n-1]
#else
    unsigned int Random_int (unsigned int n); // [0,n-1]
#endif

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    unsigned int Random_int (unsigned int n, RandomEngine const& random_engine)
    {
#if AMREX_DEVICE_COMPILE
#if defined(__SYCL_DEVICE_ONLY__)
        if (n <= static_cast<unsigned int>(std::numeric_limits<std::int32_t>::max())) {
            mkl::rng::device::uniform<std::int32_t> distr(0,static_cast<std::int32_t>(n));
            return mkl::rng::device::generate(distr, *random_engine.engine);
        } else {
            amrex::Abort("Random_int not supported in DPC++ device code, if n > INT_MAX");
            return 0; // xxxxx DPCPP todo: unsigned uniform distribution
        }
#else
        int rand;
        constexpr unsigned int RAND_M = 4294967295; // 2**32-1
        do {
            AMREX_HIP_OR_CUDA( rand = hiprand(random_engine.rand_state);,
                               rand =  curand(random_engine.rand_state) );
        } while (rand > (RAND_M - RAND_M % n));
        return rand % n;
#endif
#else
        amrex::ignore_unused(random_engine);
        return Random_int(n);
#endif
    }

#ifdef AMREX_USE_GPU
    // Locking mechanism functions for locking and unlocking
    AMREX_GPU_DEVICE int get_state (int tid);
    AMREX_GPU_DEVICE void free_state (int tid);
#endif

    /**
    * \brief Generates one pseudorandom unsigned long which is
    *  uniformly distributed on [0,n-1]-interval for each call.
    *
    * The CPU version of this function uses C++11's mt19937.
    * There is no GPU version.
    */
    AMREX_GPU_HOST ULong Random_long (ULong n); // [0,n-1]

    /** \brief Set the seed of the random number generator.
    *
    *  There is also an entry point for Fortran callable as:
    *
    *  INTEGER seed
    *  call blutilinitrand(seed)
    *
    *  or
    *
    *  INTEGER seed
    *  call blinitrand(seed)
    */
    void InitRandom (ULong seed, int nprocs=ParallelDescriptor::NProcs());

    void ResetRandomSeed (ULong seed);

    /**
    * \brief Save and restore random state.
    *
    */
    void SaveRandomState (std::ostream& os);

    void RestoreRandomState (std::istream& is, int nthreads_old, int nstep_old);

    /**
    * \brief Create a unique subset of random numbers from a pool
    *   of integers in the range [0, poolSize - 1]
    *   the set will be in the order they are found
    *   setSize must be <= poolSize
    *   uSet will be resized to setSize
    *   if you want all processors to have the same set,
    *   call this on one processor and broadcast the array
    */
    void UniqueRandomSubset (Vector<int> &uSet, int setSize, int poolSize,
                             bool printSet = false);

    /**
    * \brief Fill binCounts with the number of items for each bin.
    *   The remainder items are randomly assigned.
    */
    void NItemsPerBin (int totalItems, Vector<int> &binCounts);

    void DeallocateRandomSeedDevArray ();
}

#endif

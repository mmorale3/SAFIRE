#ifndef AFQMC_PHMSD_VMF_KERNELS_HPP
#define AFQMC_PHMSD_VMF_KERNELS_HPP

#include <complex>

namespace kernels
{
// GPU evaluation of the off-diagonal contribution to the PHMSD mean-field Green's
// function (PHMSD::vMF). Replaces the O(sum nnz^2) host loop over excitation pairs.
//
//  For each coupling row nd of OpSpinDetCouplings[other_spin] (columns = `spin` config
//  indices), scan all pairs (n1<n2) of coupled configs. If the two configs differ by a
//  single orbital (excitation number == 1, r removed / a added), accumulate:
//     G[a][:] += w        * conj(OrbMats[spin][r][:])
//     G[r][:] += conj(w)  * conj(OrbMats[spin][a][:])
//  with w = coup_val[n1] * conj(coup_val[n2]).  (sign is ignored, as in the host code.)
//
//  configs: sorted occupied-orbital lists, [nconfig][NAEA] (row-major, device).
//  coupling / OrbMats are CSR (int pointers, int column indices, complex values).
void vmf_offdiag(int nd_count,
                 int rank,
                 int size,
                 int NAEA,
                 int const* coup_pb,
                 int const* coup_pe,
                 int const* coup_jdet,
                 std::complex<double> const* coup_val,
                 int const* configs,
                 int const* orb_pb,
                 int const* orb_pe,
                 int const* orb_idx,
                 std::complex<double> const* orb_val,
                 std::complex<double>* G,
                 int Gstride);

} // namespace kernels

#endif

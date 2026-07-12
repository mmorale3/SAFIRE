#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"
#include "Numerics/detail/CUDA/Kernels/phmsd_vmf.cuh"

namespace kernels
{

__device__ __forceinline__ void atomicAddComplex(thrust::complex<double>* addr, thrust::complex<double> v)
{
  double* p = reinterpret_cast<double*>(addr);
  atomicAdd(p, v.real());
  atomicAdd(p + 1, v.imag());
}

// one thread-block per coupling row nd; block threads split the nnz*nnz pair space.
__global__ void kernel_vmf_offdiag(int nd_count,
                                   int rank,
                                   int size,
                                   int NAEA,
                                   int const* coup_pb,
                                   int const* coup_pe,
                                   int const* coup_jdet,
                                   thrust::complex<double> const* coup_val,
                                   int const* configs,
                                   int const* orb_pb,
                                   int const* orb_pe,
                                   int const* orb_idx,
                                   thrust::complex<double> const* orb_val,
                                   thrust::complex<double>* G,
                                   int Gstride)
{
  int nd = blockIdx.x;
  if (nd >= nd_count)
    return;
  if (nd % size != rank)
    return;
  int b   = coup_pb[nd];
  int e   = coup_pe[nd];
  int nnz = e - b;
  long tot = (long)nnz * (long)nnz;
  for (long idx = (long)threadIdx.x; idx < tot; idx += (long)blockDim.x)
  {
    int n1 = (int)(idx / nnz);
    int n2 = (int)(idx % nnz);
    if (n2 <= n1)
      continue;
    int const* c1 = configs + (long)coup_jdet[b + n1] * NAEA;
    int const* c2 = configs + (long)coup_jdet[b + n2] * NAEA;
    // merge two sorted configs -> removed (in c1 not c2), added (in c2 not c1)
    int ia = 0, ib = 0, nrem = 0, nadd = 0, removed = -1, added = -1;
    while (ia < NAEA && ib < NAEA)
    {
      int x = c1[ia], y = c2[ib];
      if (x == y) { ia++; ib++; }
      else if (x < y) { removed = x; nrem++; ia++; if (nrem > 1) break; }
      else { added = y; nadd++; ib++; if (nadd > 1) break; }
    }
    if (nrem > 1 || nadd > 1)
      continue;
    nrem += (NAEA - ia);
    nadd += (NAEA - ib);
    if (nrem != 1 || nadd != 1)
      continue;
    if (ia < NAEA) removed = c1[ia];
    if (ib < NAEA) added = c2[ib];
    int r = removed, a = added;
    thrust::complex<double> w = coup_val[b + n1] * thrust::conj(coup_val[b + n2]);
    thrust::complex<double> wc = thrust::conj(w);
    // G[a][:] += w * conj(OrbMats[r][:])
    for (int ip = orb_pb[r]; ip < orb_pe[r]; ip++)
      atomicAddComplex(G + (long)a * Gstride + orb_idx[ip], w * thrust::conj(orb_val[ip]));
    // G[r][:] += conj(w) * conj(OrbMats[a][:])
    for (int ip = orb_pb[a]; ip < orb_pe[a]; ip++)
      atomicAddComplex(G + (long)r * Gstride + orb_idx[ip], wc * thrust::conj(orb_val[ip]));
  }
}

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
                 int Gstride)
{
  if (nd_count <= 0)
    return;
  int nthreads = 128;
  dim3 grid(nd_count, 1, 1);
  kernel_vmf_offdiag<<<grid, nthreads>>>(
      nd_count, rank, size, NAEA, coup_pb, coup_pe, coup_jdet,
      reinterpret_cast<thrust::complex<double> const*>(coup_val), configs, orb_pb, orb_pe, orb_idx,
      reinterpret_cast<thrust::complex<double> const*>(orb_val),
      reinterpret_cast<thrust::complex<double>*>(G), Gstride);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

} // namespace kernels

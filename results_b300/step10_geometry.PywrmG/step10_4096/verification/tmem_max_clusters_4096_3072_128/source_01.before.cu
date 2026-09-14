
#include <cuda/barrier>
#include <cooperative_groups.h>


#ifdef __CUDACC_RTC__
  #include <cuda/std/cstdint>
  using cuda::std::uint8_t;
  using cuda::std::uint16_t;
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  using cuda::std::int8_t;
  using cuda::std::int16_t;
  using cuda::std::int32_t;
  using cuda::std::int64_t;

  #include <cuda/std/type_traits>
  namespace std {
    using cuda::std::is_same;
    using cuda::std::is_same_v;
    using cuda::std::is_integral;
    using cuda::std::is_signed;
    using cuda::std::is_unsigned;
    using cuda::std::is_floating_point;
    using cuda::std::enable_if;
    using cuda::std::conditional;
  }

  // NVRTC uses asm/volatile instead of __asm__/__volatile__ (gcc extension).
  #ifndef __asm__
  #define __asm__ asm
  #endif
  #ifndef __volatile__
  #define __volatile__ volatile
  #endif
#else
  #include <cstdint>
  #include <type_traits>
  #include <cuda.h>
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#endif // __CUDA_ARCH__ >= 530

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
#if ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)))
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
#endif
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

template <typename T, typename TVec2>
struct __align__(8) half4_bfloat164 {
  T x, y, z, w;
  __host__ __device__ half4_bfloat164() : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
  __host__ __device__ half4_bfloat164(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

};

using half4 = half4_bfloat164<__half, __half2>;
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4(x, y, z, w);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
#endif

__forceinline__ __device__ uint32_t get_tmem_addr(uint32_t idx, int row_offset, int col_offset) {
  int col_idx = idx & 0xFFFF;
  int row_idx = (idx >> 16) & 0xFFFF;
  col_idx += col_offset;
  row_idx += row_offset;
  col_idx = col_idx & 0xFFFF;
  row_idx = row_idx & 0xFFFF;

  uint32_t new_idx = (row_idx << 16) | col_idx;
  return new_idx;
}

#ifndef HOST_DEVICE
#define HOST_DEVICE __forceinline__ __host__ __device__
#endif
union SmemDescriptor
{
  uint64_t desc_ = 0;
  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;                     // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    uint16_t leading_byte_offset_ : 14, : 2;               // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    uint16_t stride_byte_offset_ : 14, version_ : 2;       // 14 bits [0,14), 2 bits [14,16)
    // base_offset, bit [49,52). leading_byte_offset_mode, bit [52,53).
    uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;     // 1 bit unused, 3 bits [1,4), 1 bit [4,5), 3 bits unused
    // layout type, bit [61,64), SWIZZLE_NONE matrix descriptor = 0, SWIZZLE_128B matrix descriptor = 2, SWIZZLE_64B descriptor = 4, SWIZZLE_32B descriptor = 6, SWIZZLE_128B_BASE32B = 1, N/A = 3, N/A = 5, N/A = 7
    uint8_t : 5, layout_type_ : 3;                         // 6 bits unused, 3 bits [5,8)
  };
  // Seperate the field, as we may only update one part of desc
  struct {
    uint32_t lo;
    uint32_t hi;
  };

  // Decay to a uint64_t
  HOST_DEVICE constexpr
  operator uint64_t() const noexcept { return desc_; }
};

__forceinline__ __device__ uint32_t tvm_builtin_elect_one_sync() {{
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
}}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_dealloc_cta_group_2(uint32_t taddr, int nCols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" : : "r"(taddr), "r"(nCols) : "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_2() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}

__forceinline__ __device__ void ptx_cp_async_bulk_wait_group_read_0() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__forceinline__ __device__ void ptx_cp_async_bulk_tensor_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}

__forceinline__ __device__ void tvm_builtin_cuda_cta_sync() {
    __syncthreads();
}

template <typename T>
__forceinline__ __device__ T* tvm_builtin_pointer_offset(T* ptr, int offset) {
    return ptr + offset;
}

__forceinline__ __device__ void tvm_builtin_ptx_fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_init(void* barrier, int thread_count) {
    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(barrier_addr), "r"(thread_count) : "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_st_plain_shared_v4_u32_from_src(void* address, void* src_ptr, unsigned long long cache_policy) {
    unsigned int addr = (unsigned int)__cvta_generic_to_shared(address);
    uint4 src_ = *reinterpret_cast<uint4*>(src_ptr);
    unsigned int r0 = src_.x;
    unsigned int r1 = src_.y;
    unsigned int r2 = src_.z;
    unsigned int r3 = src_.w;
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
                 : "memory");
}

__forceinline__ __device__ int tvm_builtin_cuda_thread_rank() {
    namespace cg = cooperative_groups;
    return cg::this_thread_block().thread_rank();
}

__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(uint64_t desc_base, int32_t offset) {
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}

__forceinline__ __device__ void ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr(void* dst, unsigned int mbar_addr, unsigned long long tensormap_addr, uint16_t cta_mask, unsigned long long cache_policy, int coord0, int coord1) {
    unsigned int dst_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"(dst_addr), "l"(tensormap_addr), "r"(mbar_addr),
          "r"(coord0), "r"(coord1)
        : "memory"
    );
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__forceinline__ __device__ unsigned int tvm_builtin_cluster_ctaid_y() {
  unsigned int ctaid;
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(ctaid) :);
  return ctaid;
}

__forceinline__ __device__ void ptx_cp_async_bulk_tensor_shared_to_global_2d(void* src, unsigned long long tensormap_addr, unsigned long long cache_policy, int coord0, int coord1) {
    unsigned int src_addr = __cvta_generic_to_shared(src);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :
        : "l"(tensormap_addr), "r"(src_addr),
          "r"(coord0), "r"(coord1)
        : "memory"
    );
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_alloc_cta_group_2(void* dst, int nCols) {
    unsigned int dst_addr = __cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" : : "r"(dst_addr), "r"(nCols) : "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_arrive_expect_tx_shared_cluster_remote_pred(void* barrier, int byte_count, int remote, int pred) {
    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .b32 remAddr32;\n"
        "setp.ne.s32 p, %3, 0;\n"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %2;\n"
        "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %1;\n"
        "}\n"
        :: "r"(barrier_addr), "r"(byte_count), "r"(remote), "r"(pred) : "memory");
}
__forceinline__ __device__ unsigned int tvm_builtin_cluster_ctaid_x() {
  unsigned int ctaid;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(ctaid) :);
  return ctaid;
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_ld_32x32b_x32(void* reg0, void* reg1, void* reg2, void* reg3, void* reg4, void* reg5, void* reg6, void* reg7, void* reg8, void* reg9, void* reg10, void* reg11, void* reg12, void* reg13, void* reg14, void* reg15, void* reg16, void* reg17, void* reg18, void* reg19, void* reg20, void* reg21, void* reg22, void* reg23, void* reg24, void* reg25, void* reg26, void* reg27, void* reg28, void* reg29, void* reg30, void* reg31, uint32_t taddr, uint32_t row_offset, uint32_t col_offset) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "[%32];\n"
        :  "=r"(*(uint32_t*)reg0), "=r"(*(uint32_t*)reg1), "=r"(*(uint32_t*)reg2), "=r"(*(uint32_t*)reg3), "=r"(*(uint32_t*)reg4), "=r"(*(uint32_t*)reg5), "=r"(*(uint32_t*)reg6), "=r"(*(uint32_t*)reg7), "=r"(*(uint32_t*)reg8), "=r"(*(uint32_t*)reg9), "=r"(*(uint32_t*)reg10), "=r"(*(uint32_t*)reg11), "=r"(*(uint32_t*)reg12), "=r"(*(uint32_t*)reg13), "=r"(*(uint32_t*)reg14), "=r"(*(uint32_t*)reg15), "=r"(*(uint32_t*)reg16), "=r"(*(uint32_t*)reg17), "=r"(*(uint32_t*)reg18), "=r"(*(uint32_t*)reg19), "=r"(*(uint32_t*)reg20), "=r"(*(uint32_t*)reg21), "=r"(*(uint32_t*)reg22), "=r"(*(uint32_t*)reg23), "=r"(*(uint32_t*)reg24), "=r"(*(uint32_t*)reg25), "=r"(*(uint32_t*)reg26), "=r"(*(uint32_t*)reg27), "=r"(*(uint32_t*)reg28), "=r"(*(uint32_t*)reg29), "=r"(*(uint32_t*)reg30), "=r"(*(uint32_t*)reg31)
        :  "r"(get_tmem_addr(taddr, row_offset, col_offset))
        :
    );
}

__forceinline__ __device__ unsigned int tvm_builtin_cuda_cvta_generic_to_shared(void* p) {
    return __cvta_generic_to_shared(p);
}

__forceinline__ __device__ uint32_t tvm_builtin_elect_one_sync_op() {
    return tvm_builtin_elect_one_sync();
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__forceinline__ __device__ void tvm_builtin_cuda_cluster_sync() {
    asm("barrier.cluster.arrive.aligned;");
    asm("barrier.cluster.wait.aligned;");
}

__forceinline__ __device__ void ptx_tcgen05_mma_cta_2_kind_f16_SS(uint32_t d_tmem_addr, uint64_t a_operand, uint64_t b_desc, uint32_t i_desc, uint32_t scaleC, uint32_t mask0, uint32_t mask1, uint32_t mask2, uint32_t mask3, uint32_t mask4, uint32_t mask5, uint32_t mask6, uint32_t mask7) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, "
        "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n"
        "}\n"
        :
        : "r"(d_tmem_addr), "l"(a_operand), "l"(b_desc), "r"(i_desc), "r"(scaleC), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3), "r"(mask4), "r"(mask5), "r"(mask6), "r"(mask7)
    );
}

__forceinline__ __device__ uint64_t tvm_builtin_ptx_mapa_u64(void* addr, uint32_t rank) {
    uint64_t result;
    asm volatile("mapa.u64 %0, %1, %2;"
                 : "=l"(result) : "l"(addr), "r"(rank));
    return result;
}

__forceinline__ __device__ void tvm_builtin_cast_float32x2_float16x2(void* dst, void* src) {
    ((half2*)dst)[0] = __float22half2_rn(((float2*)src)[0]);
}

__forceinline__ __device__ void tvm_builtin_ptx_fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__forceinline__ __device__ void ptx_tcgen05_commit_cta_group_2_multicast(void* bar, uint16_t cta_mask) {
    unsigned int bar_addr = __cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" : : "r"(bar_addr), "h"(cta_mask) : "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_try_wait(void* barrier, int phase) {
    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
    unsigned int ticks = 0x989680;
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_encode_matrix_descriptor(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle) {
  SmemDescriptor _desc{};  // value-init: reading uncovered pad bits is UB

  _desc.version_ = 1;
  _desc.lbo_mode_ = 0;

  switch (swizzle) {
    case 0: _desc.layout_type_ = uint8_t(0); break; // No swizzle
    case 1: _desc.layout_type_ = uint8_t(6); break; // 32B swizzle
    case 2: _desc.layout_type_ = uint8_t(4); break; // 64B swizzle
    case 3: _desc.layout_type_ = uint8_t(2); break; // 128B swizzle
    case 4: _desc.layout_type_ = uint8_t(1); break; // 128B_base32B swizzle
  }

  uint32_t start_address = __cvta_generic_to_shared(addr);
  _desc.start_address_ = static_cast<uint16_t>(start_address >> 4);

  constexpr uint8_t base_offset = 0;
  _desc.base_offset_ = base_offset;

  _desc.stride_byte_offset_  = static_cast<uint32_t>(sdo);
  _desc.leading_byte_offset_ = static_cast<uint32_t>(ldo);

  *desc = (uint64_t)_desc;
}

__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(void* barrier, int remote, int pred) {
    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .b32 remAddr32;\n"
        "setp.ne.s32 p, %2, 0;\n"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n"
        "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n"
        "}\n"
        :: "r"(barrier_addr), "r"(remote), "r"(pred) : "memory");
}

__forceinline__ __device__ void tvm_builtin_cuda_warpgroup_sync(int name_bar_id) {
    asm volatile("bar.sync %0, 128;" : : "r"(name_bar_id));
}
extern "C" __global__ void __launch_bounds__(384) kernel_kernel(const __grid_constant__ CUtensorMap A_tensormap, const __grid_constant__ CUtensorMap B_tensormap, const __grid_constant__ CUtensorMap D_tensormap);
extern "C" __global__ void __launch_bounds__(384) kernel_kernel(const __grid_constant__ CUtensorMap A_tensormap, const __grid_constant__ CUtensorMap B_tensormap, const __grid_constant__ CUtensorMap D_tensormap) {
  int warp_id_in_cta = __shfl_sync((uint)4294967295, (((int)threadIdx.x) >> 5), 0, 32);
  int bx = ((int)blockIdx.x);
  int cbx = ((int)tvm_builtin_cluster_ctaid_x());
  int cby = 0;
  int cse_v1 = (warp_id_in_cta >> 2);
  int wg_id = (warp_id_in_cta >> 2);
  int cse_v2 = (warp_id_in_cta & 3);
  int warp_id = (warp_id_in_cta & 3);
  int cse_v3 = (((int)threadIdx.x) & 31);
  int lane_id = (((int)threadIdx.x) & 31);
  extern __shared__ __align__(64) uchar pool_buf_ptr[];
  alignas(64) uint64_t descA_ptr[1];
  tvm_builtin_ptx_tcgen05_encode_matrix_descriptor((&(descA_ptr[0])), (&(((half*)pool_buf_ptr)[512])), 0, 64, 3);
  alignas(64) uint64_t descB_ptr[1];
  tvm_builtin_ptx_tcgen05_encode_matrix_descriptor((&(descB_ptr[0])), (&(((half*)pool_buf_ptr)[66048])), 0, 64, 3);
  if (tvm_builtin_cuda_thread_rank() == 0) {
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[1])), 1);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[2])), 1);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[3])), 1);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[4])), 1);
  }
  if (tvm_builtin_cuda_thread_rank() == 0) {
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[5])), 2);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[6])), 2);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[7])), 2);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[8])), 2);
  }
  if (tvm_builtin_cuda_thread_rank() == 0) {
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[9])), 1);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[10])), 1);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[11])), 1);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[12])), 1);
  }
  if (tvm_builtin_cuda_thread_rank() == 0) {
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[13])), 256);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[14])), 256);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[15])), 256);
    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[16])), 256);
  }
  if ((warp_id_in_cta >> 2) == 0) {
    if (warp_id_in_cta == 0) {
      tvm_builtin_ptx_tcgen05_alloc_cta_group_2((&(((uint*)pool_buf_ptr)[0])), 512);
    }
  }
  tvm_builtin_ptx_fence_proxy_async_shared_cta();
  tvm_builtin_ptx_fence_mbarrier_init();
  tvm_builtin_cuda_cta_sync();
  tvm_builtin_cuda_cluster_sync();
  uint mma_tmem_base = ((uint*)pool_buf_ptr)[0];
  float* tmem = ((float*)((float*)mma_tmem_base));
  alignas(64) int _ptr[1];
  alignas(64) int _ptr_1[1];
  alignas(64) int _ptr_2[1];
  alignas(64) int tile_scheduler_tile_count_ptr[1];
  int cse_v4 = (((int)blockIdx.x) >> 1);
  _ptr_2[0] = (((int)blockIdx.x) >> 1);
  tile_scheduler_tile_count_ptr[0] = 0;
  if ((bool)1 & (bool)1) {
    int group_id = 0;
    int within_group = (((int)blockIdx.x) >> 1);
    int cse_v19 = ((((int)blockIdx.x) & 15) >> 1);
    int tile_row = ((((int)blockIdx.x) & 15) >> 1);
    int cse_v5 = (((int)blockIdx.x) >> 4);
    int tile_col = (((int)blockIdx.x) >> 4);
    _ptr[0] = ((((int)blockIdx.x) & 15) >> 1);
    _ptr_1[0] = (((int)blockIdx.x) >> 4);
  } else {
    _ptr[0] = 0;
    _ptr_1[0] = 0;
  }
  uint64_t* remote_mbar_ptr = (uint64_t*)((uint64_t*)tvm_builtin_ptx_mapa_u64((&(((uint64_t*)pool_buf_ptr)[1])), 0));
  alignas(64) int tma_phase_stage_ptr[1];
  alignas(64) int tma_phase_phase_ptr[1];
  alignas(64) int mma_phase_stage_ptr[1];
  alignas(64) int mma_phase_phase_ptr[1];
  alignas(64) int ld_phase_stage_ptr[1];
  alignas(64) int ld_phase_phase_ptr[1];
  alignas(64) int wb_phase_stage_ptr[1];
  alignas(64) int wb_phase_phase_ptr[1];
  tma_phase_stage_ptr[0] = 0;
  tma_phase_phase_ptr[0] = 1;
  mma_phase_stage_ptr[0] = 0;
  mma_phase_phase_ptr[0] = 0;
  ld_phase_stage_ptr[0] = 0;
  ld_phase_phase_ptr[0] = 1;
  wb_phase_stage_ptr[0] = 0;
  wb_phase_phase_ptr[0] = 0;
  int cse_v7 = (((int)tvm_builtin_cluster_ctaid_x()) * 128);
  if ((warp_id_in_cta >> 2) == 2) {
    if ((warp_id_in_cta & 3) == 3) {
      if (tvm_builtin_elect_one_sync_op() != (uint)0) {
        #pragma unroll 1
        while (1) {
          if (!((_ptr_2[0] < 192))) { break; }
          for (int k = 0; k < 2; ++k) {
            tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[(tma_phase_stage_ptr[0] + 5)])), (tma_phase_phase_ptr[0] ^ 0));
            int cse_v6 = (k * 64);
            ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr((&(((half*)pool_buf_ptr)[((tma_phase_stage_ptr[0] * 4096) + 66048)])), tvm_builtin_cuda_cvta_generic_to_shared((&(remote_mbar_ptr[tma_phase_stage_ptr[0]]))), ((unsigned long long)(&(B_tensormap))), 0, (uint64_t)0, (k * 64), ((_ptr_1[0] * 128) + (((int)tvm_builtin_cluster_ctaid_x()) * 64)));
            ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr((&(((half*)pool_buf_ptr)[((tma_phase_stage_ptr[0] * 16384) + 512)])), tvm_builtin_cuda_cvta_generic_to_shared((&(remote_mbar_ptr[tma_phase_stage_ptr[0]]))), ((unsigned long long)(&(A_tensormap))), 0, (uint64_t)0, (k * 64), ((_ptr[0] * 512) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));
            ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr((&(((half*)pool_buf_ptr)[((tma_phase_stage_ptr[0] * 16384) + 8704)])), tvm_builtin_cuda_cvta_generic_to_shared((&(remote_mbar_ptr[tma_phase_stage_ptr[0]]))), ((unsigned long long)(&(A_tensormap))), 0, (uint64_t)0, (k * 64), (((_ptr[0] * 512) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)) + 256));
            if (((int)tvm_builtin_cluster_ctaid_x()) == 0) {
              alignas(64) bool actual_pred_ptr[1];
              actual_pred_ptr[0] = (bool)1;
              tvm_builtin_ptx_mbarrier_arrive_expect_tx_shared_cluster_remote_pred((&(((uint64_t*)pool_buf_ptr)[(tma_phase_stage_ptr[0] + 1)])), 81920, 0, actual_pred_ptr[0]);
            }
            tma_phase_stage_ptr[0] = (tma_phase_stage_ptr[0] + 1);
            if (tma_phase_stage_ptr[0] == 4) {
              tma_phase_stage_ptr[0] = 0;
              tma_phase_phase_ptr[0] = (tma_phase_phase_ptr[0] ^ 1);
            }
          }
          _ptr_2[0] = (_ptr_2[0] + 74);
          tile_scheduler_tile_count_ptr[0] = (tile_scheduler_tile_count_ptr[0] + 1);
          if ((bool)1 & (_ptr_2[0] < 192)) {
            int group_id_1 = ((_ptr_2[0] / 192) + ((_ptr_2[0] % 192) >> 31));
            int within_group_1 = ((_ptr_2[0] % 192) + (192 & ((_ptr_2[0] % 192) >> 31)));
            int cse_v20 = ((group_id_1 * 8) + (within_group_1 & 7));
            int tile_row_1 = ((group_id_1 * 8) + (within_group_1 & 7));
            int cse_v8 = (within_group_1 >> 3);
            int tile_col_1 = (within_group_1 >> 3);
            _ptr[0] = ((group_id_1 * 8) + (within_group_1 & 7));
            _ptr_1[0] = (within_group_1 >> 3);
          } else {
            _ptr[0] = 0;
            _ptr_1[0] = 0;
          }
        }
      }
    } else {
      if ((warp_id_in_cta & 3) < 2) {
        if (((int)tvm_builtin_cluster_ctaid_x()) == 0) {
          if (tvm_builtin_elect_one_sync_op() != (uint)0) {
            #pragma unroll 1
            while (1) {
              if (!((_ptr_2[0] < 192))) { break; }
              tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[(((ld_phase_stage_ptr[0] * 2) + (warp_id_in_cta & 3)) + 13)])), (ld_phase_phase_ptr[0] ^ 0));
              for (int k_1 = 0; k_1 < 2; ++k_1) {
                tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + 1)])), (mma_phase_phase_ptr[0] ^ 0));
                tvm_builtin_ptx_tcgen05_fence_after_thread_sync();
                int cse_v21 = ((warp_id_in_cta & 3) * 128);
                int cse_v22 = ((warp_id_in_cta & 3) * 1024);
                ptx_tcgen05_mma_cta_2_kind_f16_SS((mma_tmem_base + ((uint)((ld_phase_stage_ptr[0] * 256) + ((warp_id_in_cta & 3) * 128)))), tvm_builtin_smem_desc_add_16B_offset(descA_ptr[0], ((mma_phase_stage_ptr[0] * 2048) + ((warp_id_in_cta & 3) * 1024))), tvm_builtin_smem_desc_add_16B_offset(descB_ptr[0], (mma_phase_stage_ptr[0] * 512)), (uint)270532624, (0 < k_1), 0, 0, 0, 0, 0, 0, 0, 0);
                ptx_tcgen05_mma_cta_2_kind_f16_SS((mma_tmem_base + ((uint)((ld_phase_stage_ptr[0] * 256) + ((warp_id_in_cta & 3) * 128)))), tvm_builtin_smem_desc_add_16B_offset(descA_ptr[0], (((mma_phase_stage_ptr[0] * 2048) + ((warp_id_in_cta & 3) * 1024)) + 2)), tvm_builtin_smem_desc_add_16B_offset(descB_ptr[0], ((mma_phase_stage_ptr[0] * 512) + 2)), (uint)270532624, (bool)1, 0, 0, 0, 0, 0, 0, 0, 0);
                ptx_tcgen05_mma_cta_2_kind_f16_SS((mma_tmem_base + ((uint)((ld_phase_stage_ptr[0] * 256) + ((warp_id_in_cta & 3) * 128)))), tvm_builtin_smem_desc_add_16B_offset(descA_ptr[0], (((mma_phase_stage_ptr[0] * 2048) + ((warp_id_in_cta & 3) * 1024)) + 4)), tvm_builtin_smem_desc_add_16B_offset(descB_ptr[0], ((mma_phase_stage_ptr[0] * 512) + 4)), (uint)270532624, (bool)1, 0, 0, 0, 0, 0, 0, 0, 0);
                ptx_tcgen05_mma_cta_2_kind_f16_SS((mma_tmem_base + ((uint)((ld_phase_stage_ptr[0] * 256) + ((warp_id_in_cta & 3) * 128)))), tvm_builtin_smem_desc_add_16B_offset(descA_ptr[0], (((mma_phase_stage_ptr[0] * 2048) + ((warp_id_in_cta & 3) * 1024)) + 6)), tvm_builtin_smem_desc_add_16B_offset(descB_ptr[0], ((mma_phase_stage_ptr[0] * 512) + 6)), (uint)270532624, (bool)1, 0, 0, 0, 0, 0, 0, 0, 0);
                ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + 5)])), 3);
                mma_phase_stage_ptr[0] = (mma_phase_stage_ptr[0] + 1);
                if (mma_phase_stage_ptr[0] == 4) {
                  mma_phase_stage_ptr[0] = 0;
                  mma_phase_phase_ptr[0] = (mma_phase_phase_ptr[0] ^ 1);
                }
              }
              ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[(((ld_phase_stage_ptr[0] * 2) + (warp_id_in_cta & 3)) + 9)])), 3);
              ld_phase_stage_ptr[0] = (ld_phase_stage_ptr[0] + 1);
              if (ld_phase_stage_ptr[0] == 2) {
                ld_phase_stage_ptr[0] = 0;
                ld_phase_phase_ptr[0] = (ld_phase_phase_ptr[0] ^ 1);
              }
              _ptr_2[0] = (_ptr_2[0] + 74);
              tile_scheduler_tile_count_ptr[0] = (tile_scheduler_tile_count_ptr[0] + 1);
              if ((bool)1 & (_ptr_2[0] < 192)) {
                int group_id_2 = ((_ptr_2[0] / 192) + ((_ptr_2[0] % 192) >> 31));
                int within_group_2 = ((_ptr_2[0] % 192) + (192 & ((_ptr_2[0] % 192) >> 31)));
                int cse_v23 = ((group_id_2 * 8) + (within_group_2 & 7));
                int tile_row_2 = ((group_id_2 * 8) + (within_group_2 & 7));
                int cse_v9 = (within_group_2 >> 3);
                int tile_col_2 = (within_group_2 >> 3);
                _ptr[0] = ((group_id_2 * 8) + (within_group_2 & 7));
                _ptr_1[0] = (within_group_2 >> 3);
              } else {
                _ptr[0] = 0;
                _ptr_1[0] = 0;
              }
            }
          }
        }
      }
    }
  } else {
    if (warp_id_in_cta < 8) {
      alignas(64) float Dreg_ptr[32];
      alignas(64) half Dreg_f16_ptr[128];
      #pragma unroll 1
      while (1) {
        if (!((_ptr_2[0] < 192))) { break; }
        tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[(((wb_phase_stage_ptr[0] * 2) + (warp_id_in_cta >> 2)) + 9)])), (wb_phase_phase_ptr[0] ^ 0));
        tvm_builtin_ptx_tcgen05_fence_after_thread_sync();
        int cse_v24 = ((warp_id_in_cta >> 2) * 128);
        tvm_builtin_ptx_tcgen05_ld_32x32b_x32((&(Dreg_ptr[0])), (&(Dreg_ptr[1])), (&(Dreg_ptr[2])), (&(Dreg_ptr[3])), (&(Dreg_ptr[4])), (&(Dreg_ptr[5])), (&(Dreg_ptr[6])), (&(Dreg_ptr[7])), (&(Dreg_ptr[8])), (&(Dreg_ptr[9])), (&(Dreg_ptr[10])), (&(Dreg_ptr[11])), (&(Dreg_ptr[12])), (&(Dreg_ptr[13])), (&(Dreg_ptr[14])), (&(Dreg_ptr[15])), (&(Dreg_ptr[16])), (&(Dreg_ptr[17])), (&(Dreg_ptr[18])), (&(Dreg_ptr[19])), (&(Dreg_ptr[20])), (&(Dreg_ptr[21])), (&(Dreg_ptr[22])), (&(Dreg_ptr[23])), (&(Dreg_ptr[24])), (&(Dreg_ptr[25])), (&(Dreg_ptr[26])), (&(Dreg_ptr[27])), (&(Dreg_ptr[28])), (&(Dreg_ptr[29])), (&(Dreg_ptr[30])), (&(Dreg_ptr[31])), mma_tmem_base, 0, ((wb_phase_stage_ptr[0] * 256) + ((warp_id_in_cta >> 2) * 128)));
        tvm_builtin_ptx_tcgen05_wait_ld();
        for (int f = 0; f < 16; ++f) {
          alignas(64) int dst_lane_indices_0_0_ptr[1];
          int cse_v10 = (f * 2);
          dst_lane_indices_0_0_ptr[0] = (f * 2);
          alignas(64) int dst_lane_indices_1_0_ptr[1];
          dst_lane_indices_1_0_ptr[0] = ((f * 2) + 1);
          tvm_builtin_cast_float32x2_float16x2((&(Dreg_f16_ptr[(f * 2)])), (&(Dreg_ptr[(f * 2)])));
        }
        tvm_builtin_ptx_tcgen05_ld_32x32b_x32((&(Dreg_ptr[0])), (&(Dreg_ptr[1])), (&(Dreg_ptr[2])), (&(Dreg_ptr[3])), (&(Dreg_ptr[4])), (&(Dreg_ptr[5])), (&(Dreg_ptr[6])), (&(Dreg_ptr[7])), (&(Dreg_ptr[8])), (&(Dreg_ptr[9])), (&(Dreg_ptr[10])), (&(Dreg_ptr[11])), (&(Dreg_ptr[12])), (&(Dreg_ptr[13])), (&(Dreg_ptr[14])), (&(Dreg_ptr[15])), (&(Dreg_ptr[16])), (&(Dreg_ptr[17])), (&(Dreg_ptr[18])), (&(Dreg_ptr[19])), (&(Dreg_ptr[20])), (&(Dreg_ptr[21])), (&(Dreg_ptr[22])), (&(Dreg_ptr[23])), (&(Dreg_ptr[24])), (&(Dreg_ptr[25])), (&(Dreg_ptr[26])), (&(Dreg_ptr[27])), (&(Dreg_ptr[28])), (&(Dreg_ptr[29])), (&(Dreg_ptr[30])), (&(Dreg_ptr[31])), mma_tmem_base, 0, (((wb_phase_stage_ptr[0] * 256) + ((warp_id_in_cta >> 2) * 128)) + 32));
        tvm_builtin_ptx_tcgen05_wait_ld();
        for (int f_1 = 0; f_1 < 16; ++f_1) {
          alignas(64) int dst_lane_indices_0_0_ptr_1[1];
          int cse_v11 = (f_1 * 2);
          dst_lane_indices_0_0_ptr_1[0] = (f_1 * 2);
          alignas(64) int dst_lane_indices_1_0_ptr_1[1];
          dst_lane_indices_1_0_ptr_1[0] = ((f_1 * 2) + 1);
          tvm_builtin_cast_float32x2_float16x2((&(Dreg_f16_ptr[((f_1 * 2) + 32)])), (&(Dreg_ptr[(f_1 * 2)])));
        }
        tvm_builtin_ptx_tcgen05_ld_32x32b_x32((&(Dreg_ptr[0])), (&(Dreg_ptr[1])), (&(Dreg_ptr[2])), (&(Dreg_ptr[3])), (&(Dreg_ptr[4])), (&(Dreg_ptr[5])), (&(Dreg_ptr[6])), (&(Dreg_ptr[7])), (&(Dreg_ptr[8])), (&(Dreg_ptr[9])), (&(Dreg_ptr[10])), (&(Dreg_ptr[11])), (&(Dreg_ptr[12])), (&(Dreg_ptr[13])), (&(Dreg_ptr[14])), (&(Dreg_ptr[15])), (&(Dreg_ptr[16])), (&(Dreg_ptr[17])), (&(Dreg_ptr[18])), (&(Dreg_ptr[19])), (&(Dreg_ptr[20])), (&(Dreg_ptr[21])), (&(Dreg_ptr[22])), (&(Dreg_ptr[23])), (&(Dreg_ptr[24])), (&(Dreg_ptr[25])), (&(Dreg_ptr[26])), (&(Dreg_ptr[27])), (&(Dreg_ptr[28])), (&(Dreg_ptr[29])), (&(Dreg_ptr[30])), (&(Dreg_ptr[31])), mma_tmem_base, 0, (((wb_phase_stage_ptr[0] * 256) + ((warp_id_in_cta >> 2) * 128)) + 64));
        tvm_builtin_ptx_tcgen05_wait_ld();
        for (int f_2 = 0; f_2 < 16; ++f_2) {
          alignas(64) int dst_lane_indices_0_0_ptr_2[1];
          int cse_v12 = (f_2 * 2);
          dst_lane_indices_0_0_ptr_2[0] = (f_2 * 2);
          alignas(64) int dst_lane_indices_1_0_ptr_2[1];
          dst_lane_indices_1_0_ptr_2[0] = ((f_2 * 2) + 1);
          tvm_builtin_cast_float32x2_float16x2((&(Dreg_f16_ptr[((f_2 * 2) + 64)])), (&(Dreg_ptr[(f_2 * 2)])));
        }
        tvm_builtin_ptx_tcgen05_ld_32x32b_x32((&(Dreg_ptr[0])), (&(Dreg_ptr[1])), (&(Dreg_ptr[2])), (&(Dreg_ptr[3])), (&(Dreg_ptr[4])), (&(Dreg_ptr[5])), (&(Dreg_ptr[6])), (&(Dreg_ptr[7])), (&(Dreg_ptr[8])), (&(Dreg_ptr[9])), (&(Dreg_ptr[10])), (&(Dreg_ptr[11])), (&(Dreg_ptr[12])), (&(Dreg_ptr[13])), (&(Dreg_ptr[14])), (&(Dreg_ptr[15])), (&(Dreg_ptr[16])), (&(Dreg_ptr[17])), (&(Dreg_ptr[18])), (&(Dreg_ptr[19])), (&(Dreg_ptr[20])), (&(Dreg_ptr[21])), (&(Dreg_ptr[22])), (&(Dreg_ptr[23])), (&(Dreg_ptr[24])), (&(Dreg_ptr[25])), (&(Dreg_ptr[26])), (&(Dreg_ptr[27])), (&(Dreg_ptr[28])), (&(Dreg_ptr[29])), (&(Dreg_ptr[30])), (&(Dreg_ptr[31])), mma_tmem_base, 0, (((wb_phase_stage_ptr[0] * 256) + ((warp_id_in_cta >> 2) * 128)) + 96));
        tvm_builtin_ptx_tcgen05_wait_ld();
        for (int f_3 = 0; f_3 < 16; ++f_3) {
          alignas(64) int dst_lane_indices_0_0_ptr_3[1];
          int cse_v13 = (f_3 * 2);
          dst_lane_indices_0_0_ptr_3[0] = (f_3 * 2);
          alignas(64) int dst_lane_indices_1_0_ptr_3[1];
          dst_lane_indices_1_0_ptr_3[0] = ((f_3 * 2) + 1);
          tvm_builtin_cast_float32x2_float16x2((&(Dreg_f16_ptr[((f_3 * 2) + 96)])), (&(Dreg_ptr[(f_3 * 2)])));
        }
        tvm_builtin_ptx_tcgen05_fence_before_thread_sync();
        alignas(64) bool actual_pred_ptr_1[1];
        actual_pred_ptr_1[0] = (bool)1;
        tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred((&(((uint64_t*)pool_buf_ptr)[(((wb_phase_stage_ptr[0] * 2) + (warp_id_in_cta >> 2)) + 13)])), 0, actual_pred_ptr_1[0]);
        wb_phase_stage_ptr[0] = (wb_phase_stage_ptr[0] + 1);
        if (wb_phase_stage_ptr[0] == 2) {
          wb_phase_stage_ptr[0] = 0;
          wb_phase_phase_ptr[0] = (wb_phase_phase_ptr[0] ^ 1);
        }
        alignas(64) int s_off_ptr[1];
        int cse_v28 = ((warp_id_in_cta * 1024) + ((((int)threadIdx.x) & 31) * 32));
        s_off_ptr[0] = ((warp_id_in_cta * 1024) + ((((int)threadIdx.x) & 31) * 32));
        alignas(64) int _ptr_3[2];
        _ptr_3[0] = (16 - ((((s_off_ptr[0] >> 3) >> 4) & 1) * 32));
        _ptr_3[1] = (8 - ((((s_off_ptr[0] >> 3) >> 3) & 1) * 16));
        for (int f_4 = 0; f_4 < 4; ++f_4) {
          alignas(64) int ds_ptr[1];
          int cse_v14 = (f_4 * 8);
          ds_ptr[0] = (f_4 * 8);
          alignas(64) int dr_ptr[1];
          dr_ptr[0] = (f_4 * 8);
          void* s_ptr = (void*)tvm_builtin_pointer_offset((&(((half*)pool_buf_ptr)[82432])), ((((((s_off_ptr[0] >> 3) ^ (((s_off_ptr[0] >> 3) & 24) >> 3)) << 3) + (((f_4 >> 1) & 1) * _ptr_3[0])) + ((f_4 & 1) * _ptr_3[1])) + (s_off_ptr[0] & 7)));
          void* r_ptr = (void*)tvm_builtin_pointer_offset((&(Dreg_f16_ptr[0])), dr_ptr[0]);
          tvm_builtin_ptx_st_plain_shared_v4_u32_from_src(s_ptr, r_ptr, (uint64_t)0);
        }
        tvm_builtin_ptx_fence_proxy_async_shared_cta();
        int cse_v25 = ((warp_id_in_cta >> 2) + 10);
        tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));
        int cse_v26 = ((warp_id_in_cta >> 2) * 256);
        int cse_v29 = (((warp_id_in_cta >> 2) * 4096) + 82432);
        if ((warp_id_in_cta % 4) == 0) {
          if (tvm_builtin_elect_one_sync_op() != (uint)0) {
            ptx_cp_async_bulk_tensor_shared_to_global_2d((&(((half*)pool_buf_ptr)[(((warp_id_in_cta >> 2) * 4096) + 82432)])), ((unsigned long long)(&(D_tensormap))), (uint64_t)0, (_ptr_1[0] * 128), (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256)) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));
            ptx_cp_async_bulk_tensor_commit_group();
            ptx_cp_async_bulk_wait_group_read_0();
          }
        }
        tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));
        alignas(64) int s_off_ptr_1[1];
        s_off_ptr_1[0] = ((warp_id_in_cta * 1024) + ((((int)threadIdx.x) & 31) * 32));
        alignas(64) int _ptr_4[2];
        _ptr_4[0] = (16 - ((((s_off_ptr_1[0] >> 3) >> 4) & 1) * 32));
        _ptr_4[1] = (8 - ((((s_off_ptr_1[0] >> 3) >> 3) & 1) * 16));
        for (int f_5 = 0; f_5 < 4; ++f_5) {
          alignas(64) int ds_ptr_1[1];
          int cse_v15 = (f_5 * 8);
          ds_ptr_1[0] = (f_5 * 8);
          alignas(64) int dr_ptr_1[1];
          dr_ptr_1[0] = (f_5 * 8);
          void* s_ptr_1 = (void*)tvm_builtin_pointer_offset((&(((half*)pool_buf_ptr)[82432])), ((((((s_off_ptr_1[0] >> 3) ^ (((s_off_ptr_1[0] >> 3) & 24) >> 3)) << 3) + (((f_5 >> 1) & 1) * _ptr_4[0])) + ((f_5 & 1) * _ptr_4[1])) + (s_off_ptr_1[0] & 7)));
          void* r_ptr_1 = (void*)tvm_builtin_pointer_offset((&(Dreg_f16_ptr[0])), (dr_ptr_1[0] + 32));
          tvm_builtin_ptx_st_plain_shared_v4_u32_from_src(s_ptr_1, r_ptr_1, (uint64_t)0);
        }
        tvm_builtin_ptx_fence_proxy_async_shared_cta();
        tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));
        if ((warp_id_in_cta % 4) == 0) {
          if (tvm_builtin_elect_one_sync_op() != (uint)0) {
            ptx_cp_async_bulk_tensor_shared_to_global_2d((&(((half*)pool_buf_ptr)[(((warp_id_in_cta >> 2) * 4096) + 82432)])), ((unsigned long long)(&(D_tensormap))), (uint64_t)0, ((_ptr_1[0] * 128) + 32), (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256)) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));
            ptx_cp_async_bulk_tensor_commit_group();
            ptx_cp_async_bulk_wait_group_read_0();
          }
        }
        tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));
        alignas(64) int s_off_ptr_2[1];
        s_off_ptr_2[0] = ((warp_id_in_cta * 1024) + ((((int)threadIdx.x) & 31) * 32));
        alignas(64) int _ptr_5[2];
        _ptr_5[0] = (16 - ((((s_off_ptr_2[0] >> 3) >> 4) & 1) * 32));
        _ptr_5[1] = (8 - ((((s_off_ptr_2[0] >> 3) >> 3) & 1) * 16));
        for (int f_6 = 0; f_6 < 4; ++f_6) {
          alignas(64) int ds_ptr_2[1];
          int cse_v16 = (f_6 * 8);
          ds_ptr_2[0] = (f_6 * 8);
          alignas(64) int dr_ptr_2[1];
          dr_ptr_2[0] = (f_6 * 8);
          void* s_ptr_2 = (void*)tvm_builtin_pointer_offset((&(((half*)pool_buf_ptr)[82432])), ((((((s_off_ptr_2[0] >> 3) ^ (((s_off_ptr_2[0] >> 3) & 24) >> 3)) << 3) + (((f_6 >> 1) & 1) * _ptr_5[0])) + ((f_6 & 1) * _ptr_5[1])) + (s_off_ptr_2[0] & 7)));
          void* r_ptr_2 = (void*)tvm_builtin_pointer_offset((&(Dreg_f16_ptr[0])), (dr_ptr_2[0] + 64));
          tvm_builtin_ptx_st_plain_shared_v4_u32_from_src(s_ptr_2, r_ptr_2, (uint64_t)0);
        }
        tvm_builtin_ptx_fence_proxy_async_shared_cta();
        tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));
        if ((warp_id_in_cta % 4) == 0) {
          if (tvm_builtin_elect_one_sync_op() != (uint)0) {
            ptx_cp_async_bulk_tensor_shared_to_global_2d((&(((half*)pool_buf_ptr)[(((warp_id_in_cta >> 2) * 4096) + 82432)])), ((unsigned long long)(&(D_tensormap))), (uint64_t)0, ((_ptr_1[0] * 128) + 64), (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256)) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));
            ptx_cp_async_bulk_tensor_commit_group();
            ptx_cp_async_bulk_wait_group_read_0();
          }
        }
        tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));
        alignas(64) int s_off_ptr_3[1];
        s_off_ptr_3[0] = ((warp_id_in_cta * 1024) + ((((int)threadIdx.x) & 31) * 32));
        alignas(64) int _ptr_6[2];
        _ptr_6[0] = (16 - ((((s_off_ptr_3[0] >> 3) >> 4) & 1) * 32));
        _ptr_6[1] = (8 - ((((s_off_ptr_3[0] >> 3) >> 3) & 1) * 16));
        for (int f_7 = 0; f_7 < 4; ++f_7) {
          alignas(64) int ds_ptr_3[1];
          int cse_v17 = (f_7 * 8);
          ds_ptr_3[0] = (f_7 * 8);
          alignas(64) int dr_ptr_3[1];
          dr_ptr_3[0] = (f_7 * 8);
          void* s_ptr_3 = (void*)tvm_builtin_pointer_offset((&(((half*)pool_buf_ptr)[82432])), ((((((s_off_ptr_3[0] >> 3) ^ (((s_off_ptr_3[0] >> 3) & 24) >> 3)) << 3) + (((f_7 >> 1) & 1) * _ptr_6[0])) + ((f_7 & 1) * _ptr_6[1])) + (s_off_ptr_3[0] & 7)));
          void* r_ptr_3 = (void*)tvm_builtin_pointer_offset((&(Dreg_f16_ptr[0])), (dr_ptr_3[0] + 96));
          tvm_builtin_ptx_st_plain_shared_v4_u32_from_src(s_ptr_3, r_ptr_3, (uint64_t)0);
        }
        tvm_builtin_ptx_fence_proxy_async_shared_cta();
        tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));
        if ((warp_id_in_cta % 4) == 0) {
          if (tvm_builtin_elect_one_sync_op() != (uint)0) {
            ptx_cp_async_bulk_tensor_shared_to_global_2d((&(((half*)pool_buf_ptr)[(((warp_id_in_cta >> 2) * 4096) + 82432)])), ((unsigned long long)(&(D_tensormap))), (uint64_t)0, ((_ptr_1[0] * 128) + 96), (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256)) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));
            ptx_cp_async_bulk_tensor_commit_group();
            ptx_cp_async_bulk_wait_group_read_0();
          }
        }
        tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));
        _ptr_2[0] = (_ptr_2[0] + 74);
        tile_scheduler_tile_count_ptr[0] = (tile_scheduler_tile_count_ptr[0] + 1);
        if ((bool)1 & (_ptr_2[0] < 192)) {
          int group_id_3 = ((_ptr_2[0] / 192) + ((_ptr_2[0] % 192) >> 31));
          int within_group_3 = ((_ptr_2[0] % 192) + (192 & ((_ptr_2[0] % 192) >> 31)));
          int cse_v27 = ((group_id_3 * 8) + (within_group_3 & 7));
          int tile_row_3 = ((group_id_3 * 8) + (within_group_3 & 7));
          int cse_v18 = (within_group_3 >> 3);
          int tile_col_3 = (within_group_3 >> 3);
          _ptr[0] = ((group_id_3 * 8) + (within_group_3 & 7));
          _ptr_1[0] = (within_group_3 >> 3);
        } else {
          _ptr[0] = 0;
          _ptr_1[0] = 0;
        }
      }
    }
  }
  tvm_builtin_cuda_cluster_sync();
  if ((warp_id_in_cta >> 2) == 0) {
    if (warp_id_in_cta == 0) {
      tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_2();
      tvm_builtin_ptx_tcgen05_dealloc_cta_group_2(((uint*)pool_buf_ptr)[0], 512);
    }
  }
}


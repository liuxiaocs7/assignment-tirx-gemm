
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

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_dealloc_cta_group_1(uint32_t taddr, int nCols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" : : "r"(taddr), "r"(nCols) : "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_1() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__forceinline__ __device__ void ptx_cp_async_bulk_tensor_shared_to_global_3d(void* src, unsigned long long tensormap_addr, unsigned long long cache_policy, int coord0, int coord1, int coord2) {
    unsigned int src_addr = __cvta_generic_to_shared(src);
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :
        : "l"(tensormap_addr), "r"(src_addr),
          "r"(coord0), "r"(coord1), "r"(coord2)
        : "memory"
    );
}

__forceinline__ __device__ void tvm_builtin_cast_float32x2_float16x2(void* dst, void* src) {
    ((half2*)dst)[0] = __float22half2_rn(((float2*)src)[0]);
}

__forceinline__ __device__ void ptx_cp_async_bulk_wait_group_read_0() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_ld_32x32b_x128(void* reg0, void* reg1, void* reg2, void* reg3, void* reg4, void* reg5, void* reg6, void* reg7, void* reg8, void* reg9, void* reg10, void* reg11, void* reg12, void* reg13, void* reg14, void* reg15, void* reg16, void* reg17, void* reg18, void* reg19, void* reg20, void* reg21, void* reg22, void* reg23, void* reg24, void* reg25, void* reg26, void* reg27, void* reg28, void* reg29, void* reg30, void* reg31, void* reg32, void* reg33, void* reg34, void* reg35, void* reg36, void* reg37, void* reg38, void* reg39, void* reg40, void* reg41, void* reg42, void* reg43, void* reg44, void* reg45, void* reg46, void* reg47, void* reg48, void* reg49, void* reg50, void* reg51, void* reg52, void* reg53, void* reg54, void* reg55, void* reg56, void* reg57, void* reg58, void* reg59, void* reg60, void* reg61, void* reg62, void* reg63, void* reg64, void* reg65, void* reg66, void* reg67, void* reg68, void* reg69, void* reg70, void* reg71, void* reg72, void* reg73, void* reg74, void* reg75, void* reg76, void* reg77, void* reg78, void* reg79, void* reg80, void* reg81, void* reg82, void* reg83, void* reg84, void* reg85, void* reg86, void* reg87, void* reg88, void* reg89, void* reg90, void* reg91, void* reg92, void* reg93, void* reg94, void* reg95, void* reg96, void* reg97, void* reg98, void* reg99, void* reg100, void* reg101, void* reg102, void* reg103, void* reg104, void* reg105, void* reg106, void* reg107, void* reg108, void* reg109, void* reg110, void* reg111, void* reg112, void* reg113, void* reg114, void* reg115, void* reg116, void* reg117, void* reg118, void* reg119, void* reg120, void* reg121, void* reg122, void* reg123, void* reg124, void* reg125, void* reg126, void* reg127, uint32_t taddr, uint32_t row_offset, uint32_t col_offset) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x128.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
        "[%128];\n"
        :  "=r"(*(uint32_t*)reg0), "=r"(*(uint32_t*)reg1), "=r"(*(uint32_t*)reg2), "=r"(*(uint32_t*)reg3), "=r"(*(uint32_t*)reg4), "=r"(*(uint32_t*)reg5), "=r"(*(uint32_t*)reg6), "=r"(*(uint32_t*)reg7), "=r"(*(uint32_t*)reg8), "=r"(*(uint32_t*)reg9), "=r"(*(uint32_t*)reg10), "=r"(*(uint32_t*)reg11), "=r"(*(uint32_t*)reg12), "=r"(*(uint32_t*)reg13), "=r"(*(uint32_t*)reg14), "=r"(*(uint32_t*)reg15), "=r"(*(uint32_t*)reg16), "=r"(*(uint32_t*)reg17), "=r"(*(uint32_t*)reg18), "=r"(*(uint32_t*)reg19), "=r"(*(uint32_t*)reg20), "=r"(*(uint32_t*)reg21), "=r"(*(uint32_t*)reg22), "=r"(*(uint32_t*)reg23), "=r"(*(uint32_t*)reg24), "=r"(*(uint32_t*)reg25), "=r"(*(uint32_t*)reg26), "=r"(*(uint32_t*)reg27), "=r"(*(uint32_t*)reg28), "=r"(*(uint32_t*)reg29), "=r"(*(uint32_t*)reg30), "=r"(*(uint32_t*)reg31), "=r"(*(uint32_t*)reg32), "=r"(*(uint32_t*)reg33), "=r"(*(uint32_t*)reg34), "=r"(*(uint32_t*)reg35), "=r"(*(uint32_t*)reg36), "=r"(*(uint32_t*)reg37), "=r"(*(uint32_t*)reg38), "=r"(*(uint32_t*)reg39), "=r"(*(uint32_t*)reg40), "=r"(*(uint32_t*)reg41), "=r"(*(uint32_t*)reg42), "=r"(*(uint32_t*)reg43), "=r"(*(uint32_t*)reg44), "=r"(*(uint32_t*)reg45), "=r"(*(uint32_t*)reg46), "=r"(*(uint32_t*)reg47), "=r"(*(uint32_t*)reg48), "=r"(*(uint32_t*)reg49), "=r"(*(uint32_t*)reg50), "=r"(*(uint32_t*)reg51), "=r"(*(uint32_t*)reg52), "=r"(*(uint32_t*)reg53), "=r"(*(uint32_t*)reg54), "=r"(*(uint32_t*)reg55), "=r"(*(uint32_t*)reg56), "=r"(*(uint32_t*)reg57), "=r"(*(uint32_t*)reg58), "=r"(*(uint32_t*)reg59), "=r"(*(uint32_t*)reg60), "=r"(*(uint32_t*)reg61), "=r"(*(uint32_t*)reg62), "=r"(*(uint32_t*)reg63), "=r"(*(uint32_t*)reg64), "=r"(*(uint32_t*)reg65), "=r"(*(uint32_t*)reg66), "=r"(*(uint32_t*)reg67), "=r"(*(uint32_t*)reg68), "=r"(*(uint32_t*)reg69), "=r"(*(uint32_t*)reg70), "=r"(*(uint32_t*)reg71), "=r"(*(uint32_t*)reg72), "=r"(*(uint32_t*)reg73), "=r"(*(uint32_t*)reg74), "=r"(*(uint32_t*)reg75), "=r"(*(uint32_t*)reg76), "=r"(*(uint32_t*)reg77), "=r"(*(uint32_t*)reg78), "=r"(*(uint32_t*)reg79), "=r"(*(uint32_t*)reg80), "=r"(*(uint32_t*)reg81), "=r"(*(uint32_t*)reg82), "=r"(*(uint32_t*)reg83), "=r"(*(uint32_t*)reg84), "=r"(*(uint32_t*)reg85), "=r"(*(uint32_t*)reg86), "=r"(*(uint32_t*)reg87), "=r"(*(uint32_t*)reg88), "=r"(*(uint32_t*)reg89), "=r"(*(uint32_t*)reg90), "=r"(*(uint32_t*)reg91), "=r"(*(uint32_t*)reg92), "=r"(*(uint32_t*)reg93), "=r"(*(uint32_t*)reg94), "=r"(*(uint32_t*)reg95), "=r"(*(uint32_t*)reg96), "=r"(*(uint32_t*)reg97), "=r"(*(uint32_t*)reg98), "=r"(*(uint32_t*)reg99), "=r"(*(uint32_t*)reg100), "=r"(*(uint32_t*)reg101), "=r"(*(uint32_t*)reg102), "=r"(*(uint32_t*)reg103), "=r"(*(uint32_t*)reg104), "=r"(*(uint32_t*)reg105), "=r"(*(uint32_t*)reg106), "=r"(*(uint32_t*)reg107), "=r"(*(uint32_t*)reg108), "=r"(*(uint32_t*)reg109), "=r"(*(uint32_t*)reg110), "=r"(*(uint32_t*)reg111), "=r"(*(uint32_t*)reg112), "=r"(*(uint32_t*)reg113), "=r"(*(uint32_t*)reg114), "=r"(*(uint32_t*)reg115), "=r"(*(uint32_t*)reg116), "=r"(*(uint32_t*)reg117), "=r"(*(uint32_t*)reg118), "=r"(*(uint32_t*)reg119), "=r"(*(uint32_t*)reg120), "=r"(*(uint32_t*)reg121), "=r"(*(uint32_t*)reg122), "=r"(*(uint32_t*)reg123), "=r"(*(uint32_t*)reg124), "=r"(*(uint32_t*)reg125), "=r"(*(uint32_t*)reg126), "=r"(*(uint32_t*)reg127)
        :  "r"(get_tmem_addr(taddr, row_offset, col_offset))
        :
    );
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__forceinline__ __device__ void ptx_tcgen05_commit_cta_group_1(void* bar) {
    unsigned int bar_addr = __cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" : : "r"(bar_addr) : "memory");
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

__forceinline__ __device__ void ptx_cp_async_bulk_tensor_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
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

__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_init(void* barrier, int thread_count) {
    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(barrier_addr), "r"(thread_count) : "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_mbarrier_arrive_expect_tx_shared(void* barrier, int byte_count) {
    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(barrier_addr), "r"(byte_count) : "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__forceinline__ __device__ void tvm_builtin_ptx_fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

template <typename T>
__forceinline__ __device__ T* tvm_builtin_pointer_offset(T* ptr, int offset) {
    return ptr + offset;
}

__forceinline__ __device__ void tvm_builtin_cuda_cta_sync() {
    __syncthreads();
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__forceinline__ __device__ void ptx_tcgen05_mma_cta_1_kind_f16_SS(uint32_t d_tmem_addr, uint64_t a_operand, uint64_t b_desc, uint32_t i_desc, uint32_t scaleC, uint32_t mask0, uint32_t mask1, uint32_t mask2, uint32_t mask3) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
        "{%5, %6, %7, %8}, p;\n"
        "}\n"
        :
        : "r"(d_tmem_addr), "l"(a_operand), "l"(b_desc), "r"(i_desc), "r"(scaleC), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
    );
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__forceinline__ __device__ uint32_t tvm_builtin_elect_one_sync_op() {
    return tvm_builtin_elect_one_sync();
}

__forceinline__ __device__ void ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d(void* dst, void* mbar, unsigned long long tensormap_addr, uint16_t cta_mask, unsigned long long cache_policy, int coord0, int coord1) {
    unsigned int dst_addr = __cvta_generic_to_shared(dst);
    unsigned int mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"(dst_addr), "l"(tensormap_addr), "r"(mbar_addr),
          "r"(coord0), "r"(coord1)
        : "memory"
    );
}

__forceinline__ __device__ void tvm_builtin_ptx_tcgen05_alloc_cta_group_1(void* dst, int nCols) {
    unsigned int dst_addr = __cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" : : "r"(dst_addr), "r"(nCols) : "memory");
}

__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(uint64_t desc_base, int32_t offset) {
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}
extern "C" __global__ void __launch_bounds__(128) kernel_kernel(const __grid_constant__ CUtensorMap A_tensormap, const __grid_constant__ CUtensorMap A_tensormap_1, const __grid_constant__ CUtensorMap B_tensormap, const __grid_constant__ CUtensorMap B_tensormap_1, const __grid_constant__ CUtensorMap D_tensormap);
extern "C" __global__ void __launch_bounds__(128) kernel_kernel(const __grid_constant__ CUtensorMap A_tensormap, const __grid_constant__ CUtensorMap A_tensormap_1, const __grid_constant__ CUtensorMap B_tensormap, const __grid_constant__ CUtensorMap B_tensormap_1, const __grid_constant__ CUtensorMap D_tensormap) {
  int warp_id_in_cta = __shfl_sync((uint)4294967295, (((int)threadIdx.x) >> 5), 0, 32);
  int bx = ((int)blockIdx.x);
  int wg_id = (warp_id_in_cta >> 2);
  int cse_v1 = (warp_id_in_cta & 3);
  int warp_id = (warp_id_in_cta & 3);
  int cse_v2 = (((int)threadIdx.x) & 31);
  int lane_id = (((int)threadIdx.x) & 31);
  extern __shared__ __align__(64) uchar pool_buf_ptr[];
  alignas(64) uint64_t descA_ptr[1];
  tvm_builtin_ptx_tcgen05_encode_matrix_descriptor((&(descA_ptr[0])), (&(((half*)pool_buf_ptr)[512])), 0, 64, 3);
  alignas(64) uint64_t descB_ptr[1];
  tvm_builtin_ptx_tcgen05_encode_matrix_descriptor((&(descB_ptr[0])), (&(((half*)pool_buf_ptr)[16896])), 0, 64, 3);
  if ((warp_id_in_cta % 4) == 0) {
    if ((((int)threadIdx.x) % 32) == 0) {
      tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[1])), 1);
      tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[2])), 1);
      tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[3])), 1);
    }
    tvm_builtin_ptx_tcgen05_alloc_cta_group_1((&(((uint*)pool_buf_ptr)[0])), 512);
  }
  tvm_builtin_ptx_fence_proxy_async_shared_cta();
  tvm_builtin_ptx_fence_mbarrier_init();
  tvm_builtin_cuda_cta_sync();
  float* tmem = ((float*)((float*)((uint*)pool_buf_ptr)[0]));
  alignas(64) int _ptr[1];
  alignas(64) int _ptr_1[1];
  alignas(64) int _ptr_2[1];
  alignas(64) int tile_scheduler_tile_count_ptr[1];
  _ptr_2[0] = ((int)blockIdx.x);
  tile_scheduler_tile_count_ptr[0] = 0;
  if ((bool)1 & (bool)1) {
    int cse_v3 = (((int)blockIdx.x) >> 7);
    int group_id = (((int)blockIdx.x) >> 7);
    int cse_v4 = (((int)blockIdx.x) & 127);
    int within_group = (((int)blockIdx.x) & 127);
    int cse_v14 = (((((int)blockIdx.x) >> 7) * 8) + (((int)blockIdx.x) & 7));
    int tile_row = (((((int)blockIdx.x) >> 7) * 8) + (((int)blockIdx.x) & 7));
    int cse_v8 = ((((int)blockIdx.x) & 127) >> 3);
    int tile_col = ((((int)blockIdx.x) & 127) >> 3);
    _ptr[0] = (((((int)blockIdx.x) >> 7) * 8) + (((int)blockIdx.x) & 7));
    _ptr_1[0] = ((((int)blockIdx.x) & 127) >> 3);
  } else {
    _ptr[0] = 0;
    _ptr_1[0] = 0;
  }
  alignas(64) int phase_tma_ptr[2];
  alignas(64) int phase_mma_ptr[1];
  phase_tma_ptr[0] = 0;
  phase_tma_ptr[1] = 0;
  phase_mma_ptr[0] = 0;
  #pragma unroll 1
  while (1) {
    if (!((_ptr_2[0] < 256))) { break; }
    if ((warp_id_in_cta % 4) == 0) {
      if (tvm_builtin_elect_one_sync_op() != (uint)0) {
        ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d((&(((half*)pool_buf_ptr)[512])), (&(((uint64_t*)pool_buf_ptr)[1])), ((unsigned long long)(&(A_tensormap))), 0, (uint64_t)0, 0, (_ptr[0] * 128));
        ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d((&(((half*)pool_buf_ptr)[16896])), (&(((uint64_t*)pool_buf_ptr)[1])), ((unsigned long long)(&(B_tensormap))), 0, (uint64_t)0, 0, (_ptr_1[0] * 128));
        tvm_builtin_ptx_mbarrier_arrive_expect_tx_shared((&(((uint64_t*)pool_buf_ptr)[1])), 32768);
        ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d((&(((half*)pool_buf_ptr)[8704])), (&(((uint64_t*)pool_buf_ptr)[2])), ((unsigned long long)(&(A_tensormap))), 0, (uint64_t)0, 64, (_ptr[0] * 128));
        ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d((&(((half*)pool_buf_ptr)[25088])), (&(((uint64_t*)pool_buf_ptr)[2])), ((unsigned long long)(&(B_tensormap))), 0, (uint64_t)0, 64, (_ptr_1[0] * 128));
        tvm_builtin_ptx_mbarrier_arrive_expect_tx_shared((&(((uint64_t*)pool_buf_ptr)[2])), 32768);
        for (int k = 0; k < 32; ++k) {
          int cse_v5 = (k & 1);
          int cse_v9 = ((k & 1) + 1);
          tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[((k & 1) + 1)])), phase_tma_ptr[(k & 1)]);
          tvm_builtin_ptx_tcgen05_fence_after_thread_sync();
          int cse_v10 = ((k & 1) * 1024);
          ptx_tcgen05_mma_cta_1_kind_f16_SS(((uint*)pool_buf_ptr)[0], tvm_builtin_smem_desc_add_16B_offset(descA_ptr[0], ((k & 1) * 1024)), tvm_builtin_smem_desc_add_16B_offset(descB_ptr[0], ((k & 1) * 1024)), (uint)136314896, (0 < k), 0, 0, 0, 0);
          int cse_v15 = (((k & 1) * 1024) + 2);
          ptx_tcgen05_mma_cta_1_kind_f16_SS(((uint*)pool_buf_ptr)[0], tvm_builtin_smem_desc_add_16B_offset(descA_ptr[0], (((k & 1) * 1024) + 2)), tvm_builtin_smem_desc_add_16B_offset(descB_ptr[0], (((k & 1) * 1024) + 2)), (uint)136314896, (bool)1, 0, 0, 0, 0);
          int cse_v16 = (((k & 1) * 1024) + 4);
          ptx_tcgen05_mma_cta_1_kind_f16_SS(((uint*)pool_buf_ptr)[0], tvm_builtin_smem_desc_add_16B_offset(descA_ptr[0], (((k & 1) * 1024) + 4)), tvm_builtin_smem_desc_add_16B_offset(descB_ptr[0], (((k & 1) * 1024) + 4)), (uint)136314896, (bool)1, 0, 0, 0, 0);
          int cse_v17 = (((k & 1) * 1024) + 6);
          ptx_tcgen05_mma_cta_1_kind_f16_SS(((uint*)pool_buf_ptr)[0], tvm_builtin_smem_desc_add_16B_offset(descA_ptr[0], (((k & 1) * 1024) + 6)), tvm_builtin_smem_desc_add_16B_offset(descB_ptr[0], (((k & 1) * 1024) + 6)), (uint)136314896, (bool)1, 0, 0, 0, 0);
          ptx_tcgen05_commit_cta_group_1((&(((uint64_t*)pool_buf_ptr)[3])));
          tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[3])), phase_mma_ptr[0]);
          tvm_builtin_ptx_tcgen05_fence_after_thread_sync();
          tvm_builtin_ptx_tcgen05_fence_before_thread_sync();
          phase_tma_ptr[(k & 1)] = (phase_tma_ptr[(k & 1)] ^ 1);
          phase_mma_ptr[0] = (phase_mma_ptr[0] ^ 1);
          if (k < 30) {
            int cse_v11 = ((k & 1) * 8192);
            int cse_v12 = ((k * 64) + 128);
            ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d((&(((half*)pool_buf_ptr)[(((k & 1) * 8192) + 512)])), (&(((uint64_t*)pool_buf_ptr)[((k & 1) + 1)])), ((unsigned long long)(&(A_tensormap_1))), 0, (uint64_t)0, ((k * 64) + 128), (_ptr[0] * 128));
            ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d((&(((half*)pool_buf_ptr)[(((k & 1) * 8192) + 16896)])), (&(((uint64_t*)pool_buf_ptr)[((k & 1) + 1)])), ((unsigned long long)(&(B_tensormap_1))), 0, (uint64_t)0, ((k * 64) + 128), (_ptr_1[0] * 128));
            tvm_builtin_ptx_mbarrier_arrive_expect_tx_shared((&(((uint64_t*)pool_buf_ptr)[((k & 1) + 1)])), 32768);
          }
        }
      }
    }
    tvm_builtin_cuda_cta_sync();
    tvm_builtin_ptx_tcgen05_fence_after_thread_sync();
    alignas(64) float Dreg_ptr[128];
    alignas(64) half Dreg_f16_ptr[128];
    tvm_builtin_ptx_tcgen05_ld_32x32b_x128((&(Dreg_ptr[0])), (&(Dreg_ptr[1])), (&(Dreg_ptr[2])), (&(Dreg_ptr[3])), (&(Dreg_ptr[4])), (&(Dreg_ptr[5])), (&(Dreg_ptr[6])), (&(Dreg_ptr[7])), (&(Dreg_ptr[8])), (&(Dreg_ptr[9])), (&(Dreg_ptr[10])), (&(Dreg_ptr[11])), (&(Dreg_ptr[12])), (&(Dreg_ptr[13])), (&(Dreg_ptr[14])), (&(Dreg_ptr[15])), (&(Dreg_ptr[16])), (&(Dreg_ptr[17])), (&(Dreg_ptr[18])), (&(Dreg_ptr[19])), (&(Dreg_ptr[20])), (&(Dreg_ptr[21])), (&(Dreg_ptr[22])), (&(Dreg_ptr[23])), (&(Dreg_ptr[24])), (&(Dreg_ptr[25])), (&(Dreg_ptr[26])), (&(Dreg_ptr[27])), (&(Dreg_ptr[28])), (&(Dreg_ptr[29])), (&(Dreg_ptr[30])), (&(Dreg_ptr[31])), (&(Dreg_ptr[32])), (&(Dreg_ptr[33])), (&(Dreg_ptr[34])), (&(Dreg_ptr[35])), (&(Dreg_ptr[36])), (&(Dreg_ptr[37])), (&(Dreg_ptr[38])), (&(Dreg_ptr[39])), (&(Dreg_ptr[40])), (&(Dreg_ptr[41])), (&(Dreg_ptr[42])), (&(Dreg_ptr[43])), (&(Dreg_ptr[44])), (&(Dreg_ptr[45])), (&(Dreg_ptr[46])), (&(Dreg_ptr[47])), (&(Dreg_ptr[48])), (&(Dreg_ptr[49])), (&(Dreg_ptr[50])), (&(Dreg_ptr[51])), (&(Dreg_ptr[52])), (&(Dreg_ptr[53])), (&(Dreg_ptr[54])), (&(Dreg_ptr[55])), (&(Dreg_ptr[56])), (&(Dreg_ptr[57])), (&(Dreg_ptr[58])), (&(Dreg_ptr[59])), (&(Dreg_ptr[60])), (&(Dreg_ptr[61])), (&(Dreg_ptr[62])), (&(Dreg_ptr[63])), (&(Dreg_ptr[64])), (&(Dreg_ptr[65])), (&(Dreg_ptr[66])), (&(Dreg_ptr[67])), (&(Dreg_ptr[68])), (&(Dreg_ptr[69])), (&(Dreg_ptr[70])), (&(Dreg_ptr[71])), (&(Dreg_ptr[72])), (&(Dreg_ptr[73])), (&(Dreg_ptr[74])), (&(Dreg_ptr[75])), (&(Dreg_ptr[76])), (&(Dreg_ptr[77])), (&(Dreg_ptr[78])), (&(Dreg_ptr[79])), (&(Dreg_ptr[80])), (&(Dreg_ptr[81])), (&(Dreg_ptr[82])), (&(Dreg_ptr[83])), (&(Dreg_ptr[84])), (&(Dreg_ptr[85])), (&(Dreg_ptr[86])), (&(Dreg_ptr[87])), (&(Dreg_ptr[88])), (&(Dreg_ptr[89])), (&(Dreg_ptr[90])), (&(Dreg_ptr[91])), (&(Dreg_ptr[92])), (&(Dreg_ptr[93])), (&(Dreg_ptr[94])), (&(Dreg_ptr[95])), (&(Dreg_ptr[96])), (&(Dreg_ptr[97])), (&(Dreg_ptr[98])), (&(Dreg_ptr[99])), (&(Dreg_ptr[100])), (&(Dreg_ptr[101])), (&(Dreg_ptr[102])), (&(Dreg_ptr[103])), (&(Dreg_ptr[104])), (&(Dreg_ptr[105])), (&(Dreg_ptr[106])), (&(Dreg_ptr[107])), (&(Dreg_ptr[108])), (&(Dreg_ptr[109])), (&(Dreg_ptr[110])), (&(Dreg_ptr[111])), (&(Dreg_ptr[112])), (&(Dreg_ptr[113])), (&(Dreg_ptr[114])), (&(Dreg_ptr[115])), (&(Dreg_ptr[116])), (&(Dreg_ptr[117])), (&(Dreg_ptr[118])), (&(Dreg_ptr[119])), (&(Dreg_ptr[120])), (&(Dreg_ptr[121])), (&(Dreg_ptr[122])), (&(Dreg_ptr[123])), (&(Dreg_ptr[124])), (&(Dreg_ptr[125])), (&(Dreg_ptr[126])), (&(Dreg_ptr[127])), ((uint*)pool_buf_ptr)[0], 0, 0);
    tvm_builtin_ptx_tcgen05_wait_ld();
    tvm_builtin_ptx_tcgen05_fence_before_thread_sync();
    for (int f = 0; f < 64; ++f) {
      alignas(64) int dst_lane_indices_0_0_ptr[1];
      int cse_v6 = (f * 2);
      dst_lane_indices_0_0_ptr[0] = (f * 2);
      alignas(64) int dst_lane_indices_1_0_ptr[1];
      dst_lane_indices_1_0_ptr[0] = ((f * 2) + 1);
      tvm_builtin_cast_float32x2_float16x2((&(Dreg_f16_ptr[(f * 2)])), (&(Dreg_ptr[(f * 2)])));
    }
    alignas(64) int s_off_ptr[1];
    s_off_ptr[0] = (((warp_id_in_cta & 3) * 2048) + ((((int)threadIdx.x) & 31) * 64));
    for (int f_1 = 0; f_1 < 16; ++f_1) {
      alignas(64) int ds_ptr[1];
      ds_ptr[0] = (((f_1 >> 3) * 8192) + ((f_1 & 7) * 8));
      alignas(64) int dr_ptr[1];
      dr_ptr[0] = (f_1 * 8);
      void* s_ptr = (void*)tvm_builtin_pointer_offset((&(((half*)pool_buf_ptr)[33280])), (((((s_off_ptr[0] + ds_ptr[0]) >> 3) ^ ((((s_off_ptr[0] + ds_ptr[0]) >> 3) & 56) >> 3)) << 3) + ((s_off_ptr[0] + ds_ptr[0]) & 7)));
      void* r_ptr = (void*)tvm_builtin_pointer_offset((&(Dreg_f16_ptr[0])), dr_ptr[0]);
      tvm_builtin_ptx_st_plain_shared_v4_u32_from_src(s_ptr, r_ptr, (uint64_t)0);
    }
    tvm_builtin_ptx_fence_proxy_async_shared_cta();
    tvm_builtin_cuda_cta_sync();
    if ((warp_id_in_cta % 4) == 0) {
      if (tvm_builtin_elect_one_sync_op() != (uint)0) {
        ptx_cp_async_bulk_tensor_shared_to_global_3d((&(((half*)pool_buf_ptr)[33280])), ((unsigned long long)(&(D_tensormap))), (uint64_t)0, 0, (_ptr[0] * 128), (_ptr_1[0] * 2));
        ptx_cp_async_bulk_tensor_commit_group();
        ptx_cp_async_bulk_wait_group_read_0();
      }
    }
    tvm_builtin_cuda_cta_sync();
    _ptr_2[0] = (_ptr_2[0] + 148);
    tile_scheduler_tile_count_ptr[0] = (tile_scheduler_tile_count_ptr[0] + 1);
    if ((bool)1 & (_ptr_2[0] < 256)) {
      int group_id_1 = (_ptr_2[0] >> 7);
      int within_group_1 = (_ptr_2[0] & 127);
      int cse_v13 = ((group_id_1 * 8) + (within_group_1 & 7));
      int tile_row_1 = ((group_id_1 * 8) + (within_group_1 & 7));
      int cse_v7 = (within_group_1 >> 3);
      int tile_col_1 = (within_group_1 >> 3);
      _ptr[0] = ((group_id_1 * 8) + (within_group_1 & 7));
      _ptr_1[0] = (within_group_1 >> 3);
    } else {
      _ptr[0] = 0;
      _ptr_1[0] = 0;
    }
  }
  tvm_builtin_cuda_cta_sync();
  if ((warp_id_in_cta % 4) == 0) {
    tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_1();
    tvm_builtin_ptx_tcgen05_dealloc_cta_group_1(((uint*)pool_buf_ptr)[0], 512);
  }
}


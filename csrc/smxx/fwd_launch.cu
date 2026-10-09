#include <type_traits>

#include "fwd.h"
#include "fwd_kernel1.cuh"
#include "fwd_kernel2.cuh"

namespace {

constexpr int CHUNK = 16;

// Workspace arrays written by the prepare kernel and read by the recurrence
// kernel, in their order in the workspace buffer.
struct Workspace {
    cutlass::bfloat16_t* kd;
    cutlass::bfloat16_t* qd;
    cutlass::bfloat16_t* kr;
    float* gt;
    cutlass::bfloat16_t* inv;
    cutlass::bfloat16_t* mqk;
};

template <int D>
Workspace split_workspace(void* workspace_ptr, int H, int total_tiles) {
    using BF16 = cutlass::bfloat16_t;
    using K1L = K1Layouts<D, CHUNK>;
    using K2L = K2Layouts<D, CHUNK>;
    using WS = WorkspaceSizes<CHUNK, D>;

    // Raw bulk copies make the swizzled shared-memory layout a private ABI
    // between K1 and K2. Fail at compile time if either side changes alone.
    static_assert(std::is_same_v<typename K1L::MMALayout,
                                 typename K2L::MMALayout>);
    static_assert(std::is_same_v<typename K1L::GTotalLayout,
                                 typename K2L::GTotalLayout>);
    static_assert(std::is_same_v<typename K1L::LMLayout,
                                 typename K2L::LMLayout>);

    int64_t n_ht = int64_t(H) * total_tiles;
    char* ws = reinterpret_cast<char*>(workspace_ptr);
    return {
        reinterpret_cast<BF16*>(ws),
        reinterpret_cast<BF16*>(ws + n_ht * WS::kKDecayed),
        reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed)),
        reinterpret_cast<float*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored)),
        reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal)),
        reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal + WS::kINV)),
    };
}

}  // namespace

// ==================== launch_prepare ====================
template <int D, bool IsVarlen, typename SeqlenT>
void launch_prepare(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    float scale,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    SeqlenT const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    cudaStream_t stream
) {
#if BLOCK_LEVEL_K1 >= 0
    using BF16 = cutlass::bfloat16_t;
    using K1L = K1Layouts<D, CHUNK>;
    using TMAQKLayout = typename K1L::TMAQKLayout;
    using TMABetaSmemLayout = typename K1L::TMABetaSmemLayout;
    using TMAGTotalSmemLayout = typename K1L::TMAGTotalSmemLayout;

    auto gmem_layout = make_layout(make_shape(H, T_total, D), make_stride(D, D * H, 1));
    // 1D beta layout: [H*T] contiguous
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));
    auto [ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk] =
        split_workspace<D>(workspace_ptr, H, total_tiles);

    Tensor m_q    = make_tensor(make_gmem_ptr(q_ptr), gmem_layout);
    Tensor m_k    = make_tensor(make_gmem_ptr(k_ptr), gmem_layout);
    Tensor m_beta = make_tensor(make_gmem_ptr<BF16>(beta_ptr), beta_gmem_layout);
    auto tma_load_q    = make_tma_copy(SM90_TMA_LOAD{}, m_q, TMAQKLayout{});
    auto tma_load_k    = make_tma_copy(SM90_TMA_LOAD{}, m_k, TMAQKLayout{});
    auto tma_load_beta = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});

    Tensor m_g = make_tensor(make_gmem_ptr(g_bf16_ptr), gmem_layout);
    auto tma_load_g = make_tma_copy(SM90_TMA_LOAD{}, m_g, TMAQKLayout{});

    auto dt_bias_gmem_layout = make_layout(make_shape(H, D), LayoutRight{});
    Tensor m_dt_bias = make_tensor(make_gmem_ptr(dt_bias_ptr), dt_bias_gmem_layout);
    auto tma_load_dt_bias = make_tma_copy(SM90_TMA_LOAD{}, m_dt_bias, TMAGTotalSmemLayout{});

    constexpr int kK1Threads = 128;
    using SharedStorageK1T = SharedStorageK1<K1L>;
    int smem_size_k1 = sizeof(SharedStorageK1T);

    auto kernel1 = _flash_kda_fwd_prepare<
        decltype(tma_load_q), decltype(tma_load_k),
        decltype(tma_load_beta),
        decltype(tma_load_g), decltype(tma_load_dt_bias),
        CHUNK, D, kK1Threads, IsVarlen, SeqlenT
    >;

    cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k1);

    dim3 grid_k1(total_tiles, H);
    dim3 block_k1(kK1Threads);

    kernel1<<<grid_k1, block_k1, smem_size_k1, stream>>>(
        tma_load_q, tma_load_k, tma_load_beta,
        tma_load_g, tma_load_dt_bias,
        scale, T_total, H, N, cu_seqlens_ptr, total_tiles,
        A_log_ptr, gate_scale,
        ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk
    );
#endif
}

// ==================== launch_recurrence ====================
template <
    int D,
    bool HasStateIn,
    bool HasStateOut,
    bool StateFP32,
    bool HasCheckpoint,
    bool IsVarlen,
    typename SeqlenT>
void launch_recurrence(
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    SeqlenT const* cu_seqlens_ptr,
    int num_sms,
    cudaStream_t stream
) {
#if BLOCK_LEVEL_K2 >= 0
    using BF16 = cutlass::bfloat16_t;
    constexpr int kInputStages = 3;
    constexpr int kOutputStages = 2;
    constexpr int VD = 64;

    using K2L = K2Layouts<D, CHUNK>;
    using K2VSplitL = K2Layouts<D, CHUNK, VD>;
    using TMABetaSmemLayout = typename K1Layouts<D, CHUNK>::TMABetaSmemLayout;
    using TMAVOLayout = typename K2L::TMAVOLayout;
    using TMAStateSmemLayout = typename K2L::TMAStateSmemLayout;
    using TMAFP32StateSmemLayout = typename K2L::TMAFP32StateSmemLayout;
    using K2VSplitTMAVOLayout = typename K2VSplitL::TMAVOLayout;
    using K2VSplitTMAStateSmemLayout =
        typename K2VSplitL::TMAStateSmemLayout;
    using K2VSplitTMAFP32StateSmemLayout =
        typename K2VSplitL::TMAFP32StateSmemLayout;

    auto gmem_layout = make_layout(make_shape(H, T_total, D), make_stride(D, D * H, 1));
    // 1D beta layout: [H*T] contiguous
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));
    auto state_gmem_layout = make_layout(make_shape(N * H, D, D), LayoutRight{});
    auto [ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk] =
        split_workspace<D>(workspace_ptr, H, total_tiles);

    // --- State TMA descriptors (conditional on HasStateIn/HasStateOut and StateFP32)
    auto make_state_tma = [&](auto state_smem_layout,
                              auto fp32_state_smem_layout) {
        if constexpr (StateFP32) {
            // FP32 state TMA descriptors
            auto m_initial_fp32 = make_tensor(
                make_gmem_ptr(static_cast<float const*>(initial_state_ptr)), state_gmem_layout);
            auto m_final_fp32 = make_tensor(
                make_gmem_ptr(static_cast<float*>(final_state_ptr)), state_gmem_layout);
            auto tma_load = make_tma_copy(
                SM90_TMA_LOAD{}, m_initial_fp32, fp32_state_smem_layout);
            auto tma_store = make_tma_copy(
                SM90_TMA_STORE{}, m_final_fp32, fp32_state_smem_layout);
            return cute::make_tuple(tma_load, tma_store);
        } else {
            // BF16 state TMA descriptors (or dummy for no-state)
            auto state_ptr_load = HasStateIn
                ? static_cast<BF16 const*>(initial_state_ptr)
                : reinterpret_cast<BF16 const*>(out_ptr);  // dummy, never used
            auto state_ptr_store = HasStateOut
                ? static_cast<BF16*>(final_state_ptr)
                : reinterpret_cast<BF16*>(out_ptr);  // dummy, never used
            auto m_init = make_tensor(make_gmem_ptr(state_ptr_load), state_gmem_layout);
            auto m_final = make_tensor(make_gmem_ptr(state_ptr_store), state_gmem_layout);
            auto tma_load = make_tma_copy(
                SM90_TMA_LOAD{}, m_init, state_smem_layout);
            auto tma_store = make_tma_copy(
                SM90_TMA_STORE{}, m_final, state_smem_layout);
            return cute::make_tuple(tma_load, tma_store);
        }
    };

    // TMA descriptors for Kernel 2 inputs and outputs. Workspace payloads
    // use the raw bulk-copy pointers above rather than tensor maps.
    Tensor m_v    = make_tensor(make_gmem_ptr(v_ptr), gmem_layout);
    Tensor m_out  = make_tensor(make_gmem_ptr(out_ptr), gmem_layout);
    Tensor m_beta = make_tensor(make_gmem_ptr<BF16>(beta_ptr), beta_gmem_layout);
    auto tma_load_v     = make_tma_copy(SM90_TMA_LOAD{}, m_v, TMAVOLayout{});
    auto tma_load_beta2 = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});
    auto tma_store_out = make_tma_copy(SM90_TMA_STORE{}, m_out, TMAVOLayout{});
    auto [tma_load_initial_state, tma_store_final_state] = make_state_tma(
        TMAStateSmemLayout{}, TMAFP32StateSmemLayout{});

    // Full width: one math warpgroup plus one load/store warpgroup, for
    // register reallocation. V-split: math warpgroup plus load/store warps.
    constexpr int kK2Threads = 128 + 128;
    constexpr int kK2VSplitThreads = 32 * 2 + 128;
    dim3 block_k2(kK2Threads);

    // V-split doubles the blocks per head. Use it while its grid fits in
    // one wave.
    const int vsplit_blocks = H * (D / VD) * N;
    if (vsplit_blocks <= 2 * num_sms) {
        // Keep the default path's host launch overhead unchanged: split
        // TensorMaps are constructed only when this path may be used.
        auto tma_load_v_vsplit = make_tma_copy(
            SM90_TMA_LOAD{}, m_v, K2VSplitTMAVOLayout{});
        auto tma_store_out_vsplit = make_tma_copy(
            SM90_TMA_STORE{}, m_out, K2VSplitTMAVOLayout{});
        auto [tma_load_initial_state_vsplit,
              tma_store_final_state_vsplit] =
            make_state_tma(K2VSplitTMAStateSmemLayout{},
                           K2VSplitTMAFP32StateSmemLayout{});

        using SharedStorageK2T = SharedStorageK2<
            K2VSplitL, kInputStages, kOutputStages>;
        int smem_size_k2 = sizeof(SharedStorageK2T);
        dim3 grid_k2(H * (D / VD), N);

        auto kernel2 = _flash_kda_fwd_recurrence<
            decltype(tma_load_v_vsplit),
            decltype(tma_load_beta2),
            decltype(tma_load_initial_state_vsplit),
            decltype(tma_store_final_state_vsplit),
            decltype(tma_store_out_vsplit),
            CHUNK, D, kInputStages, kOutputStages, kK2VSplitThreads,
            HasStateIn, HasStateOut, StateFP32, HasCheckpoint,
            IsVarlen, SeqlenT,
            VD>;

        cudaFuncSetAttribute(
            kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size_k2);
        int blocks_per_sm = 1;
        // On sm_10x a V-split grid at two blocks per SM is slower than full width.
        cudaFuncAttributes attr;
        if (vsplit_blocks > num_sms &&
            cudaFuncGetAttributes(&attr, kernel2) == cudaSuccess &&
            attr.binaryVersion / 10 != 10) {
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, kernel2, kK2VSplitThreads, smem_size_k2);
        }
        if (vsplit_blocks <= blocks_per_sm * num_sms) {
            kernel2<<<grid_k2, dim3(kK2VSplitThreads), smem_size_k2, stream>>>(
                tma_load_v_vsplit, tma_load_beta2,
                tma_load_initial_state_vsplit,
                tma_store_final_state_vsplit,
                tma_store_out_vsplit,
                out_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr,
                T_total, H, N, cu_seqlens_ptr, total_tiles,
                ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk);
            return;
        }
    }

    using SharedStorageK2T = SharedStorageK2<K2L, kInputStages, kOutputStages>;
    int smem_size_k2 = sizeof(SharedStorageK2T);

    auto kernel2 = _flash_kda_fwd_recurrence<
        decltype(tma_load_v), decltype(tma_load_beta2),
        decltype(tma_load_initial_state),
        decltype(tma_store_final_state),
        decltype(tma_store_out),
        CHUNK, D, kInputStages, kOutputStages, kK2Threads,
        HasStateIn, HasStateOut, StateFP32, HasCheckpoint,
        IsVarlen, SeqlenT
    >;

    cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k2);

    // K2 maps x to head so all heads of a sequence launch together.
    // Varlen reverses y in-kernel to process vLLM's trailing prefills first.
    dim3 grid_k2(H, N);

    kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
        tma_load_v, tma_load_beta2,
        tma_load_initial_state,
        tma_store_final_state,
        tma_store_out,
        out_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr,
        T_total, H, N, cu_seqlens_ptr, total_tiles,
        ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk
    );
#endif
}

// ==================== launch_fwd ====================
template <
    int D,
    bool HasStateIn,
    bool HasStateOut,
    bool StateFP32,
    bool HasCheckpoint,
    bool IsVarlen,
    typename SeqlenT>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    SeqlenT const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    int num_sms,
    cudaStream_t stream
) {
    launch_prepare<D, IsVarlen, SeqlenT>(
        q_ptr, k_ptr, g_bf16_ptr, beta_ptr, scale, workspace_ptr,
        total_tiles, T_total, H, N, cu_seqlens_ptr, A_log_ptr, dt_bias_ptr,
        gate_scale, stream);
    launch_recurrence<D, HasStateIn, HasStateOut, StateFP32, HasCheckpoint,
                      IsVarlen, SeqlenT>(
        v_ptr, beta_ptr, initial_state_ptr, final_state_ptr,
        checkpoint_state_ptr, checkpoint_offsets_ptr, out_ptr, workspace_ptr,
        total_tiles, T_total, H, N, cu_seqlens_ptr, num_sms, stream);
}

// Explicit instantiations
#define INSTANTIATE_LAUNCH_FWD(D, HI, HO, FP32, CKPT, VL, SEQLEN_T) \
    template void launch_fwd<D, HI, HO, FP32, CKPT, VL, SEQLEN_T>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, void const*, float, void*, \
        void*, SEQLEN_T const*, cutlass::bfloat16_t*, void*, \
        int, int, int, int, \
        SEQLEN_T const*, float const*, float const*, float, int, \
        cudaStream_t);

#define INSTANTIATE_LAUNCH_RECURRENCE(D, HI, HO, FP32, CKPT, VL, SEQLEN_T) \
    template void launch_recurrence<D, HI, HO, FP32, CKPT, VL, SEQLEN_T>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        void const*, void*, void*, SEQLEN_T const*, \
        cutlass::bfloat16_t*, void*, int, int, int, int, \
        SEQLEN_T const*, int, cudaStream_t);

#define INSTANTIATE_CHECKPOINT_VARIANTS(HI, HO, FP32, VL, SEQLEN_T) \
    INSTANTIATE_LAUNCH_FWD(128, HI, HO, FP32, false, VL, SEQLEN_T) \
    INSTANTIATE_LAUNCH_FWD(128, HI, HO, FP32, true,  VL, SEQLEN_T) \
    INSTANTIATE_LAUNCH_RECURRENCE(128, HI, HO, FP32, false, VL, SEQLEN_T) \
    INSTANTIATE_LAUNCH_RECURRENCE(128, HI, HO, FP32, true,  VL, SEQLEN_T)

#define INSTANTIATE_LAUNCH_PREPARE(VL, SEQLEN_T) \
    template void launch_prepare<128, VL, SEQLEN_T>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, float, \
        void*, int, int, int, int, SEQLEN_T const*, float const*, \
        float const*, float, cudaStream_t);

#define INSTANTIATE_STATE_VARIANTS(VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(true,  true,  false, VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(true,  true,  true,  VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(false, false, false, VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(false, false, true,  VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(false, true,  false, VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(true,  false, false, VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(false, true,  true,  VL, SEQLEN_T) \
    INSTANTIATE_CHECKPOINT_VARIANTS(true,  false, true,  VL, SEQLEN_T)

INSTANTIATE_STATE_VARIANTS(true, int32_t)
INSTANTIATE_STATE_VARIANTS(true, int64_t)
INSTANTIATE_STATE_VARIANTS(false, int32_t)
INSTANTIATE_STATE_VARIANTS(false, int64_t)
INSTANTIATE_LAUNCH_PREPARE(true, int32_t)
INSTANTIATE_LAUNCH_PREPARE(true, int64_t)
INSTANTIATE_LAUNCH_PREPARE(false, int32_t)
INSTANTIATE_LAUNCH_PREPARE(false, int64_t)

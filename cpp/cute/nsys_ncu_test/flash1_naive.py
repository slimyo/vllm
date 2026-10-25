# ============================================================================
# T7 · Flash Attention Forward（Online Softmax，mma.sync 版，目标 sm_120 / 5070 Ti）
#   O = softmax(Q @ K^T * scale) @ V
#
# 说明：
#   - 5070 Ti 是消费级 Blackwell (sm_120)，没有 Hopper/B100 的 wgmma + TMA bulk
#     async copy 路径，所以沿用 mma.sync warp-level MMA（16x8x16）+ 同步 copy
#     是正确选择。
#   - 本文件把原骨架里缺失/错误的部分补全，注释里标出【原代码 bug】和
#     【调试修复 #N】方便对照调试历史。
#
# 调试修复索引（本版本已全部落地）：
#   #1  host 侧真正按 (b,h) 切片 + (B,H,S,d)->(B*H,S,d) reshape，kernel 内 2D tile
#   #2  sV 保持行主序 (BN,D)，GEMM2 的 B=V^T 用 smem 转置视图表达，不做物理转置
#   #3  GEMM1/GEMM2 各自 tiled_mma + 各自 partition
#   #4  补齐 S fragment（tCrS）的声明（经占位 smem partition 出正确 rank）
#   #5  softmax 运行统计量 m/l 全部 fp32
#   #6  P 的 fp32->fp16 不再依赖不存在的 make_fragment API，直接寄存器写 smem
#   #7  边界：本版本仍假设 S/S_kv 整除 BM/BN（测试用例保证）
#   #8  make_tiled_copy_tv 参数语义修正：thr_layout 覆盖全部线程，
#       val_layout 是"每线程"的值布局；否则 128-bit 向量化检查必炸
#       （'cannot vectorize copy to 8 elements (static strides must be 1)'）
#   #9  atom layout 改 (4,1,1)：每个 warp 独占 16 行 x 全部 64 列，
#       行统计量的 quad 级 shfl 归约才成立（同一行绝不能跨 warp）
#   #10 mma 累加器 fragment 布局：每 lane 持 2 行，每 4 个连续元素
#       [4j,4j+1]=rowA、[4j+2,4j+3]=rowB，softmax/O/rescale/epilogue 全按此分组
#   #11 S2R 拷贝用 16-bit 原子（fragment 每线程只有 2 个连续 f16，
#       装不下 128-bit 的 8 个元素，与 #8 同源的 vectorization 约束）
# ============================================================================

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

BM = 64          # Q 行 block
BN = 64          # K/V 列 block（内层循环步长）
D  = 64          # head_dim

NUM_THREADS = 128   # = 4 warp。【调试修复 #9】必须与 tiled_mma 的 atom layout
                    # (4,1,1) 对应：4 个 warp 沿 M 排布，每 warp 独占 16 行。

NEG_INF = 1e30   # fp32 安全的 -inf 替代（避免 exp(-inf - (-inf)) = nan 的边界坑）


# ============================================================================
# Warp 行归约：16x8x16 mma atom 的 C fragment 里，一个 warp 负责 16 行 x 8 列，
# 每个 lane 持有 2 行 x 2 列 = 4 个元素（见【调试修复 #10】的分组约定）。
# 同一"行"分布在 4 个 lane 上（lane l/4 == 行 / 1，即 lane 分组的 quad），
# 用 shfl_xor 在 {1,2} 两级做蝶形归约即可覆盖这 4 个 lane。
#
# 【调试修复 #9 的前提】这个 quad 级归约成立的前提是：一个 warp 独自覆盖
# 某行的"全部 64 列"。atom layout 若含 N 方向 >1 的分量（如 (4,2,1)），
# 同一行的列会被拆到多个 warp，shfl 够不到别的 warp —— 统计量只算了半行，
# 表现为 err≈0.2 量级的"少归一化了一部分 key"的温和偏差（softmax 是归一
# 化的，所以不是 NaN 也不是全错，极具迷惑性）。
# ============================================================================
@cute.jit
def warp_reduce_row_max(val: cutlass.Float32) -> cutlass.Float32:
    # 蝶形 xor 归约：lane 分组宽度 1 -> 2（一行内的 4 个 lane 恰好
    # 用两级 xor(1)/xor(2) 覆盖）
    for offset in (1, 2):
        other = cute.arch.shuffle_sync_bfly(val, offset=offset)
        val = max(val, other)
    return val


@cute.jit
def warp_reduce_row_sum(val: cutlass.Float32) -> cutlass.Float32:
    for offset in (1, 2):
        other = cute.arch.shuffle_sync_bfly(val, offset=offset)
        val = val + other
    return val


# ============================================================================
# Kernel：grid = (cdiv(S, BM), B * H)
#   blockIdx.x → Q 行 block
#   blockIdx.y → 展平后的 (batch, head) 索引
#
# 【原代码 bug #1 修正】gQ/gK/gV/gO 在 host 侧真正按 (B*H, S, d) 展平，
#   kernel 用 bidy 索引展平后的 batch-head 维，再对 (S,d) 做 2D tile。
# ============================================================================
@cute.kernel
def flash_attn_fwd_kernel(
    mQ: cute.Tensor,            # (BH, S,    d)
    mK: cute.Tensor,            # (BH, S_kv, d)
    mV: cute.Tensor,            # (BH, S_kv, d)
    mO: cute.Tensor,            # (BH, S,    d)
    scale: cutlass.Float32,
):
    # ---------- 0. 索引与块坐标 ----------
    bidx, bidy, _ = cute.arch.block_idx()   # bidx: Q 行块; bidy: 展平 (b,h)
    tidx, _, _ = cute.arch.thread_idx()

    # 先取出当前 (b,h) 对应的 2D 切片，再做 2D tiling —— 这样维度才一致。
    # 注意：Tensor.__getitem__ 的坐标只接受"整数"或"None"
    # （None = 该维整体保留），不支持 python 的 slice(":") 语法。
    gQ_bh = mQ[bidy, None, None]            # (S, d)
    gK_bh = mK[bidy, None, None]            # (S_kv, d)
    gV_bh = mV[bidy, None, None]            # (S_kv, d)
    gO_bh = mO[bidy, None, None]            # (S, d)

    gQ = cute.local_tile(gQ_bh, (BM, D), (bidx, 0))   # 本 CTA 的 Q block
    gO = cute.local_tile(gO_bh, (BM, D), (bidx, 0))   # 本 CTA 的 O block
    # gK / gV 不在这里整块切，沿 N 维流式取 —— 放到主循环里按 n_blk 切

    # ---------- 1. Shared Memory 布局 ----------
    # sQ: (BM, D) row-major，Q 加载一次后常驻
    # sK: (BN, D) row-major。【调试修复】原 sK 写成 (D,BN):(1,D) 的转置布局，
    #     与 gK (BN,D) 形状不一致，cute.copy 直接报形状错误。行主序原样存
    #     即可 —— 这恰好是 GEMM1 的 B 操作数 (N,K)=(BN,D) 的 k-major 布局，
    #     mma.sync 对 B 的 k 连续布局原生友好。
    # sV: (BN, D) row-major。【原代码 bug #2 修正】V 天然是 (BN,d)，不做物理
    #     转置；GEMM2 需要的 B=V^T (D,BN) 用"转置视图"表达（见 6.4），
    #     S2R 用 16-bit 单元素原子，任何 stride 都合法，正确性优先。
    # sP: (BM, BN) fp16，P 的 fp32->fp16 中转【原代码 bug #6 修正】
    # sS: (BM, BN) fp32，纯脚手架：只为 partition_C 出正确 rank 的 tCrS
    #     【原代码 bug #4 修正】——传 (BM,BN) tuple 给 make_fragment_C 得到
    #     的是扁平形状，和 partition 出来的 3 维 mma fragment rank 对不上。
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BM, D), stride=(D, 1)))
    sK = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BN, D), stride=(D, 1)))
    sV = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BN, D), stride=(D, 1)))
    sP = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BM, BN), stride=(BN, 1)))
    sS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BM, BN), stride=(BN, 1)))

    # ---------- 2. TiledMMA ----------
    # 基础 atom：mma.sync.m16n8k16.f32.f16.f16.f32（Ampere 起的标准 warp MMA）
    mma_atom = cute.make_mma_atom(
        cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.Float16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16),
        )
    )
    # 【调试修复 #9】atom layout (4,1,1)：
    #   逻辑 tile = (4*16, 1*8, 16) = (64, 8, 16)
    #   → M 方向 4 个 warp 覆盖 64 行（每 warp 16 行）
    #   → N 方向单个 warp，靠 rest_n = 64/8 = 8 次 atom 迭代覆盖全部 64 列
    #   → 每个 warp 独占 16 行 x 全部 64 列，行统计量不跨 warp，
    #     quad 级 shfl 归约（warp_reduce_row_*）才成立。
    # 每线程 accumulator = 8列 x 2行 = 16 元素/atom x 8 次 n 迭代 = 128？不：
    # 实际每线程持有 (MMA=4, MMA_M=1, MMA_N=8) = 32 个 fp32 累加值，
    # 4 warp x 32 = 128 lane-值 组，128 线程 x 32 = 4096 = 64x64，无冗余全覆盖。
    # 【原代码 bug #3 修正】GEMM1 的 K 维是 head_dim D，GEMM2 的 K 维是 BN，
    #   两者 (M,N,K) 语义不同；虽然此处两者 tile 恰好同为 (64,64)，仍各自
    #   make_tiled_mma + 各自 partition，语义清晰且避免共享 slice 状态。
    tiled_mma_1 = cute.make_tiled_mma(mma_atom, (4, 1, 1))   # S = Q@K^T, K维=D
    tiled_mma_2 = cute.make_tiled_mma(mma_atom, (4, 1, 1))   # O += P@V,  K维=BN

    thr_mma_1 = tiled_mma_1.get_slice(tidx)
    thr_mma_2 = tiled_mma_2.get_slice(tidx)

    # ---------- 3. 拷贝原子 ----------
    # 【调试修复 #8】make_tiled_copy_tv(atom, thr_layout, val_layout) 的参数
    #   语义：thr_layout = 参与搬数的线程布局（必须覆盖整个 CTA 的线程数）；
    #         val_layout = 每个线程搬的值布局（不是整个 tile！）。
    #   向量化不变量：每线程最内维 value 数 = num_bits / dtype_bits，
    #   且在 src/dst 上都落在 stride=1 的维度。
    #   G2S：128 bit / 16 bit = 每线程内维 8 个连续 f16。
    #   128 线程 = (32,4)，每线程 (1,8) → tile (32*1, 4*8) = (32,32)，
    #   64x64 tile 由 rest 迭代补满。
    # 【调试修复 #11】S2R：A/B fragment 每线程只有 2 个连续 f16（k 方向），
    #   装不下 8 个 —— 128-bit atom 必炸同一个 vectorization 错误。
    #   用 16-bit 原子（每条指令 1 元素，任何布局合法）；k-major 操作数
    #   以后可优化到 lmatrix / 32-bit，正确性版本先保守。
    copy_atom_g2s = cute.make_copy_atom(
        op=cute.nvgpu.CopyUniversalOp(),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=128,
    )
    copy_atom_s2r = cute.make_copy_atom(
        op=cute.nvgpu.CopyUniversalOp(),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=16,
    )

    tiled_copy_g2s_q = cute.make_tiled_copy_tv(
        copy_atom_g2s, cute.make_layout((32, 4)), cute.make_layout((1, 8)))
    tiled_copy_g2s_kv = cute.make_tiled_copy_tv(
        copy_atom_g2s, cute.make_layout((32, 4)), cute.make_layout((1, 8)))

    # ---------- 4. Q 加载：G2S -> S2R，全程常驻 ----------
    thr_copy_q = tiled_copy_g2s_q.get_slice(tidx)
    cute.copy(tiled_copy_g2s_q, thr_copy_q.partition_S(gQ), thr_copy_q.partition_D(sQ))
    cute.arch.sync_threads()

    # S2R：按 mma 的 A 操作数视角 partition sQ，读进寄存器 fragment tCrQ
    tCrQ = thr_mma_1.make_fragment_A(thr_mma_1.partition_A(sQ))
    tiled_copy_s2r_q = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma_1)
    thr_copy_s2r_q = tiled_copy_s2r_q.get_slice(tidx)
    cute.copy(tiled_copy_s2r_q,
              thr_copy_s2r_q.partition_S(sQ),
              thr_copy_s2r_q.retile(tCrQ))
    cute.arch.sync_threads()

    # ---------- 5. GEMM1 的 S fragment / GEMM2 的 O accumulator ----------
    # 【原代码 bug #4 修正】经占位 smem sS partition_C 再 make_fragment_C，
    #   拿到正确的 (atom_shape, rest_m, rest_n) 三维 fragment 结构。
    tCrS = thr_mma_1.make_fragment_C(thr_mma_1.partition_C(sS))

    # tCrO：GEMM2 的 fp32 累加器，直接对全局 gO partition（只用作结构，
    #   数值全程留寄存器，epilogue 时逐元素 cast 写回）
    tCrO = thr_mma_2.make_fragment_C(thr_mma_2.partition_C(gO))
    tCrO.fill(0.0)

    # ---------- 6. Online Softmax 状态 ----------
    # 【原代码 bug #5 修正】m/l 必须是 fp32。
    # 【调试修复 #10】mma 累加器的 canonical 布局：每个 lane 持有 2 行，
    #   每 4 个连续 fragment 元素为一组：[4j,4j+1] 属于 rowA，[4j+2,4j+3]
    #   属于 rowB（rowB = rowA + 8）。因此每线程维护两套 running stats。
    m_prev_a = cutlass.Float32(-NEG_INF)
    m_prev_b = cutlass.Float32(-NEG_INF)
    l_prev_a = cutlass.Float32(0.0)
    l_prev_b = cutlass.Float32(0.0)

    n_tiles = (gK_bh.shape[0] + BN - 1) // BN   # 手写 cdiv，不依赖版本差异

    # ---------- 7. 主循环：沿 N 维扫描 K/V ----------
    for n_blk in cutlass.range(n_tiles):
        gK = cute.local_tile(gK_bh, (BN, D), (n_blk, 0))
        gV = cute.local_tile(gV_bh, (BN, D), (n_blk, 0))

        # 7.1 K/V 流式加载 G2S（同步版；生产版改进方向：cp.async 2-stage）
        thr_copy_kv = tiled_copy_g2s_kv.get_slice(tidx)
        cute.copy(tiled_copy_g2s_kv,
                  thr_copy_kv.partition_S(gK), thr_copy_kv.partition_D(sK))
        cute.copy(tiled_copy_g2s_kv,
                  thr_copy_kv.partition_S(gV), thr_copy_kv.partition_D(sV))
        cute.arch.sync_threads()

        # 7.2 GEMM1: S = Q @ K^T * scale
        #     sK 是 (BN,D) 行主序 = B 操作数 (N,K) 的 k-major，直接 partition
        tCrK = thr_mma_1.make_fragment_B(thr_mma_1.partition_B(sK))
        cpy_s2r_k = cute.make_tiled_copy_B(copy_atom_s2r, tiled_mma_1)
        thr_s2r_k = cpy_s2r_k.get_slice(tidx)
        cute.copy(cpy_s2r_k, thr_s2r_k.partition_S(sK), thr_s2r_k.retile(tCrK))

        tCrS.fill(0.0)
        cute.gemm(tiled_mma_1, tCrS, tCrQ, tCrK, tCrS)
        for i in range(cute.size(tCrS)):
            tCrS[i] = tCrS[i] * scale

        # 7.3 Online softmax 状态机（按 rowA/rowB 两套独立统计）
        # 7.3.1 行内局部 max：每 4 元素一组的 [4j,4j+1] 归入 rowA，
        #       [4j+2,4j+3] 归入 rowB【调试修复 #10 —— 原版把两行 max 到
        #       一个标量上，统计量全错，这是 err≈0.2 的直接来源之一】
        row_max_a = cutlass.Float32(-NEG_INF)
        row_max_b = cutlass.Float32(-NEG_INF)
        for j in range(cute.size(tCrS) // 4):
            row_max_a = max(row_max_a,
                            cutlass.Float32(tCrS[4 * j + 0]),
                            cutlass.Float32(tCrS[4 * j + 1]))
            row_max_b = max(row_max_b,
                            cutlass.Float32(tCrS[4 * j + 2]),
                            cutlass.Float32(tCrS[4 * j + 3]))
        # quad 内蝶形归约：同一行的 4 个 lane 恰好持同一行的不同列
        # 【调试修复 #9 成立的前提：每 warp 独占全部 64 列】
        m_new_a = max(m_prev_a, warp_reduce_row_max(row_max_a))
        m_new_b = max(m_prev_b, warp_reduce_row_max(row_max_b))
        # cute.math.max(m_prev_a - m_new_a)
        alpha_a = cute.arch.exp(m_prev_a - m_new_a)
        alpha_b = cute.arch.exp(m_prev_b - m_new_b)

        # 7.3.2 用 alpha 修正旧的 O 累加值（online softmax 的 rescale 步骤，
        #       同样必须按行分组 —— 原版整块乘同一个 alpha 是错的）
        for j in range(cute.size(tCrO) // 4):
            tCrO[4 * j + 0] = tCrO[4 * j + 0] * alpha_a
            tCrO[4 * j + 1] = tCrO[4 * j + 1] * alpha_a
            tCrO[4 * j + 2] = tCrO[4 * j + 2] * alpha_b
            tCrO[4 * j + 3] = tCrO[4 * j + 3] * alpha_b

        # 7.3.3 exp(S - m_new) 得到未归一化权重 P，就地改写 tCrS；
        #       同步累加行和（rowA/rowB 分开）
        row_sum_a = cutlass.Float32(0.0)
        row_sum_b = cutlass.Float32(0.0)
        for j in range(cute.size(tCrS) // 4):
            pa0 = cute.arch.exp(cutlass.Float32(tCrS[4 * j + 0]) - m_new_a)
            pa1 = cute.arch.exp(cutlass.Float32(tCrS[4 * j + 1]) - m_new_a)
            pb0 = cute.arch.exp(cutlass.Float32(tCrS[4 * j + 2]) - m_new_b)
            pb1 = cute.arch.exp(cutlass.Float32(tCrS[4 * j + 3]) - m_new_b)
            tCrS[4 * j + 0] = pa0
            tCrS[4 * j + 1] = pa1
            tCrS[4 * j + 2] = pb0
            tCrS[4 * j + 3] = pb1
            row_sum_a = row_sum_a + pa0 + pa1
            row_sum_b = row_sum_b + pb0 + pb1
        l_prev_a = alpha_a * l_prev_a + warp_reduce_row_sum(row_sum_a)
        l_prev_b = alpha_b * l_prev_b + warp_reduce_row_sum(row_sum_b)
        m_prev_a = m_new_a
        m_prev_b = m_new_b

        # 7.4 P 降级到 fp16 并写入 smem
        # 【原代码 bug #6 修正】不依赖不存在的 make_fragment API；
        #   也不再走 fp32 smem 中转 —— tCrS 与 partition_C(sP) 结构同构
        #   （都是 tiled_mma_1 视角的 (64,64) C 布局），寄存器坐标与 smem
        #   坐标一一对应，直接逐元素 downcast 写入 sP。
        tCsP_w = thr_mma_1.partition_C(sP)
        for i in range(cute.size(tCrS)):
            tCsP_w[i] = cutlass.Float16(tCrS[i])
        cute.arch.sync_threads()   # 写 sP 完成，读之前必须可见

        # 7.5 GEMM2: O += P @ V
        #     A = P：从 sP 按 mma A 视角读回寄存器（16-bit S2R）
        tCrP = thr_mma_2.make_fragment_A(thr_mma_2.partition_A(sP))
        cpy_s2r_p = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma_2)
        thr_s2r_p = cpy_s2r_p.get_slice(tidx)
        cute.copy(cpy_s2r_p, thr_s2r_p.partition_S(sP), thr_s2r_p.retile(tCrP))

        #     B = V^T (D,BN)：【原代码 bug #2 修正】sV 物理上是 (BN,D)
        #     行主序，这里用 iterator + 转置 layout 造一个零拷贝的转置视图，
        #     S2R 走 16-bit 原子（任何 stride 合法，正确性优先）。
        #     生产版优化方向：物理转置 + ldmatrix.trans，或 cp.async 时转置写。
        sVT = cute.make_tensor(sV.iterator, cute.make_layout((D, BN), stride=(1, D)))
        tCrV = thr_mma_2.make_fragment_B(thr_mma_2.partition_B(sVT))
        cpy_s2r_v = cute.make_tiled_copy_B(copy_atom_s2r, tiled_mma_2)
        thr_s2r_v = cpy_s2r_v.get_slice(tidx)
        cute.copy(cpy_s2r_v, thr_s2r_v.partition_S(sVT), thr_s2r_v.retile(tCrV))

        cute.gemm(tiled_mma_2, tCrO, tCrP, tCrV, tCrO)
        cute.arch.sync_threads()   # 下一轮覆盖 sK/sV/sP 前必须同步

    # ---------- 8. 收尾：归一化 + 写回 ----------
    # 【原代码 bug #7】边界情形（S/S_kv 不整除）仍假设整除，测试用例保证；
    #   生产版必须加 predicate 逐元素 mask。
    inv_l_a = cutlass.Float32(1.0) / l_prev_a
    inv_l_b = cutlass.Float32(1.0) / l_prev_b
    # tCrO 是 fp32 accumulator，gO 是 fp16：cute.copy 不做隐式类型转换，
    # 逐元素 cast 写回【原代码 bug #7 追加修正】
    for j in range(cute.size(tCrO) // 4):
        tCgO = thr_mma_2.partition_C(gO)   # 每次引用取最新视图（编译期常量，无开销）
        tCgO[4 * j + 0] = cutlass.Float16(tCrO[4 * j + 0] * inv_l_a)
        tCgO[4 * j + 1] = cutlass.Float16(tCrO[4 * j + 1] * inv_l_a)
        tCgO[4 * j + 2] = cutlass.Float16(tCrO[4 * j + 2] * inv_l_b)
        tCgO[4 * j + 3] = cutlass.Float16(tCrO[4 * j + 3] * inv_l_b)


# ============================================================================
# Host 侧 JIT 入口
# ============================================================================
@cute.jit
def flash_attn_fwd(
    mQ: cute.Tensor,     # (BH, S,    d)
    mK: cute.Tensor,     # (BH, S_kv, d)
    mV: cute.Tensor,     # (BH, S_kv, d)
    mO: cute.Tensor,     # (BH, S,    d)
    scale: cutlass.Float32,
):
    BH, S, d = mQ.shape
    # 手写向上取整（cute.cdiv 在部分版本不存在，避免 API 演进差异）
    grid = ((S + BM - 1) // BM, BH, 1)
    # 【调试修复 #9】必须与 kernel 内 atom layout (4,1,1) 对应：4 warp
    block = (NUM_THREADS, 1, 1)
    flash_attn_fwd_kernel(mQ, mK, mV, mO, scale).launch(grid=grid, block=block)


# ============================================================================
# Python 封装：torch.Tensor -> cute.Tensor
# ============================================================================
_compiled_cache = {}

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    B, H, S, d = Q.shape
    Skv = K.shape[2]
    assert d == D, f"当前 kernel 硬编码 head_dim={D}，输入是 {d}"
    assert S % BM == 0 and Skv % BN == 0, \
        "当前版本假设 S % BM == 0 且 S_kv % BN == 0（边界 predicate 未实现）"

    O = torch.empty_like(Q)

    # 【原代码 bug #1 的落地处】(B,H,S,d) -> (B*H,S,d) 展平，
    # kernel 用 bidy 索引展平后的 batch-head 维。
    Qf = Q.reshape(B * H, S, d).contiguous()
    Kf = K.reshape(B * H, Skv, d).contiguous()
    Vf = V.reshape(B * H, Skv, d).contiguous()
    Of = O.reshape(B * H, S, d)

    mQ = from_dlpack(Qf, assumed_align=16)
    mK = from_dlpack(Kf, assumed_align=16)
    mV = from_dlpack(Vf, assumed_align=16)
    mO = from_dlpack(Of, assumed_align=16)

    scale = cutlass.Float32(1.0 / (d ** 0.5))

    key = (B * H, S, Skv, d)
    if key not in _compiled_cache:
        _compiled_cache[key] = cute.compile(flash_attn_fwd, mQ, mK, mV, mO, scale)
    _compiled_cache[key](mQ, mK, mV, mO, scale)

    return Of.reshape(B, H, S, d)


# ============================================================================
# 测试
# ============================================================================
def ref_attention(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


def make_inputs(B, H, S, Skv, d, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Q = (torch.randn(B, H, S, d, device="cuda", generator=gen) * 0.5).half()
    K = (torch.randn(B, H, Skv, d, device="cuda", generator=gen) * 0.5).half()
    V = (torch.randn(B, H, Skv, d, device="cuda", generator=gen) * 0.5).half()
    return Q, K, V


ATOL = 2e-2

TEST_SHAPES = [
    (1, 1, 64, 64, 64),
    (1, 2, 128, 128, 64),
    (2, 4, 256, 256, 64),
    (2, 8, 512, 512, 64),
    (1, 2, 128, 256, 64),
]


def test_correctness():
    print(f"{'shape':<28} {'max_abs_err':<14} {'result'}")
    print("-" * 48)
    for i, (B, H, S, Skv, d) in enumerate(TEST_SHAPES):
        Q, K, V = make_inputs(B, H, S, Skv, d, seed=i)
        O_ref = ref_attention(Q, K, V)
        O = flash_attention_forward(Q, K, V)
        assert O.shape == O_ref.shape and O.dtype == O_ref.dtype
        assert not torch.isnan(O).any(), f"[{i}] NaN in output"
        err = (O.float() - O_ref.float()).abs().max().item()
        ok = err < ATOL
        print(f"({B},{H},{S},{Skv},{d}){'':<16} {err:<14.6f} {'✅ PASS' if ok else '❌ FAIL'}")
        assert ok, f"shape {i} failed: err={err}"
    print("-" * 48)
    print("All correctness tests passed ✅")


def test_perf():
    import time
    B, H, S, d = 2, 16, 2048, 64
    Q, K, V = make_inputs(B, H, S, S, d, seed=42)
    flash_attention_forward(Q, K, V)
    torch.cuda.synchronize()
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        flash_attention_forward(Q, K, V)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
    flops = 4 * B * H * S * S * d
    print(f"[perf] {B}x{H}x{S}x{d}: {dt*1e3:.3f} ms, {flops/dt/1e12:.2f} TFLOPS")

if __name__ == "__main__":
    test_correctness()
    # test_perf()

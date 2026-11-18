# ============================================================================
# T7-async-v2 · Flash Attention Forward（Online Softmax + cp.async 4-stage
#             流水线 + ldmatrix S2R + smem swizzle，目标 sm_120 / 5070 Ti）
#   O = softmax(Q @ K^T * scale) @ V
#
# 相对 T7-async（2-stage 双缓冲版）的改动：
#
#   【优化 #6】S2R 全部从逐元素 16-bit CopyUniversalOp 换成 ldmatrix：
#       - A 操作数（Q、P）：LdMatrix8x8x16bOp x4 —— mma m16n8k16 的 A fragment
#         是 16x16 fp16 = 4 个 8x8 矩阵，一条 ldmatrix.x4 全读出来；
#       - B 操作数（K、V）：x2 —— B fragment 是 (N=8, K=16) = 2 个 8x8 矩阵；
#       - V 特殊：gemm2 的 B 是 (N=D, K=BN)，而 sV 物理上是 (BN, D) d 连续，
#         即 N 连续存储 —— 不是 mma 的原生朝向，必须用 transpose=True
#         （ldmatrix.trans，lane 地址指向 smem 里 16B 连续的行，寄存器映射
#         做转置）。这一步同时干掉了原来"裸指针转置视图 + 逐元素读"的绕法
#         的性能部分，但逻辑视图 sVT 还是要保留（见【坑 #E】）。
#       num_bits_per_copy 跟着 fragment 变：x4 -> 128bit/线程，x2 -> 64bit/线程。
#   【优化 #7】sK/sV/sP 加 128B swizzle（Swizzle<3,4,3>，fp16 行 64 元素=128B
#       的标准 swizzle）。这三块 smem 是 mma/ldmatrix 的热路径读方，
#       纯行主序下 mma.sync 访问模式撞 bank 的概率极高。sQ 只被读一次
#       （加载进 tCrQ 后全程驻留寄存器），swizzle 无收益，不加。
#   【优化 #8】cp.async 流水线 2 stage -> 4 stage：
#       - prologue 从"预取 1 块"推广为"预取 NSTAGES-1 块"（不足则能发几块发几块）；
#       - 主循环里改为"发 tile n_blk+NSTAGES-1，然后 cp_async_wait_group(NSTAGES-1)"
#         ——发完后在途共 NSTAGES 组，只等到剩 NSTAGES-1 组，即最老的
#         tile n_blk 那组落地，新发的 NSTAGES-1 组继续在途跟计算重叠；
#       - 尾段（没有下一块可发）只能 cp_async_wait_group(0) 排空。注意
#         cp.async.wait_group 的 N 是 PTX 立即数，不能传 n_tiles - n_blk - 1
#         这种运行时值；要想尾段也保留重叠，得按 constexpr 手工展开
#         NSTAGES-1 个分支，收益递减，这里不做。
#       - WAR 分析同 #4，只是推广：iter j 的 prefetch 写 stage (j+NSTAGES-1)%NSTAGES
#         = (j-1)%NSTAGES，即 iter j-1 刚用过的那块；iter j-1 结尾那个
#         sync_threads() 保证所有线程对它的 S2R 读已完成。NSTAGES 取 2 的
#         幂时 n_blk % NSTAGES 可被编译器降成位与，无除法开销。
#
# 【新踩的坑】
#   - 【坑 #E，数据正确性级别的坑】sV 加了 swizzle 之后，原来的转置视图
#       sVT = make_tensor(sV.iterator, make_layout((D, BN), stride=(1, D)))
#       是【错的】：make_tensor 用裸 stride 重建布局，swizzle 属性被绕过，
#       地址不再过 XOR —— 症状是结果整个错掉但不报任何错（比 IR
#       verification failed 难查多了）。修法：转置视图必须把同一个 swizzle
#       compose 回去（composition(sw, (D,BN):(1,D)) 在 (n,k) 处的 offset 是
#       sw(k*BN... = sw(k*64+n)，与 sV 在 (k,n) 处完全一致，数学上可验）。
#       —— 教训：swizzle 是布局属性，make_tensor 裸重建 = 静默丢属性，
#       和 #D 的 align 属于同一族问题，只不过这次不报错。
#   - ldmatrix 的 num_matrix_per_instr 要按 fragment 大小选：A 是 16x16
#       取 4，B 是 8x16 取 2。一开始 A/B 都写 4，make_tiled_copy_B 的
#       value layout 对不上 mma fragment，verification 报 shape mismatch。
#   - ldmatrix 要求每个 lane 给出的行地址 16B 对齐：sK/sV/sP 分配处
#       byte_alignment=16（#D 的教训直接沿用，这次一步到位）。
#
# 【还没实测过，仍是潜在风险点】
#   - smem 预算：NSTAGES=4 后 sK+sV = 2 * 4*64*64*2B = 64KB，加 sQ 8KB、
#     sP 8KB、sS 16KB，共 96KB。sm_120 单 block 动态 smem 上限约 99KB
#     （需要 opt-in），96KB 是贴着上限过的。如果 launch 报 smem 不足，
#     先把 NSTAGES 降到 3（总 80KB），或把 sS 改成 fp16 占位省 8KB。
#   - swizzle 布局过 make_tiled_copy_A/B 的 partition_S 兼容性检查依赖
#     DSL 的 layout 代数；万一 verification 报错，先把 swizzle 降成
#     64B 版（Swizzle<2,3,3>... 即 make_swizzle 对应组合）排查。
# ============================================================================

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

BM = 64          # Q 行 block
BN = 64          # K/V 列 block（内层循环步长）
D  = 64          # head_dim
NSTAGES = 4      # 【优化 #8】cp.async 流水线深度（2 的幂，% 降级为位与）

NUM_THREADS = 128   # = 4 warp，跟 tiled_mma 的 atom layout (4,1,1) 对应

NEG_INF = 1e30


@cute.jit
def warp_reduce_row_max(val: cutlass.Float32) -> cutlass.Float32:
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


@cute.kernel
def flash_attn_fwd_kernel(
    mQ: cute.Tensor,            # (BH, S,    d)
    mK: cute.Tensor,            # (BH, S_kv, d)
    mV: cute.Tensor,            # (BH, S_kv, d)
    mO: cute.Tensor,            # (BH, S,    d)
    scale: cutlass.Float32,
):
    # ---------- 0. 索引与块坐标 ----------
    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    gQ_bh = mQ[bidy, None, None]
    gK_bh = mK[bidy, None, None]
    gV_bh = mV[bidy, None, None]
    gO_bh = mO[bidy, None, None]

    gQ = cute.local_tile(gQ_bh, (BM, D), (bidx, 0))
    gO = cute.local_tile(gO_bh, (BM, D), (bidx, 0))

    # ---------- 1. Shared Memory 布局 ----------
    # 【优化 #7】sK/sV/sP 加 Swizzle<3,4,3>（fp16、行 64 元素 = 128B 的
    # 标准 128B swizzle）。swizzle 走 allocate_tensor 的 swizzle 参数，
    # 不在 make_layout 上手工 compose —— 分配器和布局属性一处说清。
    sw128 = cute.make_swizzle(3, 4, 3)

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((BM, D), stride=(D, 1)),
        byte_alignment=16,   # ldmatrix 要求 128-bit 对齐；之前逐元素读不需要，
                             # 换 ldsm 后必须补上（坑 #D 的第三次现身）
    )

    sK_buf = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((NSTAGES, BN, D), stride=(BN * D, D, 1)),
        byte_alignment=16,   # 【坑 #D 沿用】ldmatrix / cp.async 都要 16B 对齐
        swizzle=sw128,       # 【优化 #7】
    )
    sV_buf = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((NSTAGES, BN, D), stride=(BN * D, D, 1)),
        byte_alignment=16,
        swizzle=sw128,
    )
    sP = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((BM, BN), stride=(BN, 1)),
        byte_alignment=16,
        swizzle=sw128,
    )
    sS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BM, BN), stride=(BN, 1)))

    # ---------- 2. TiledMMA（不变） ----------
    mma_atom = cute.make_mma_atom(
        cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.Float16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16),
        )
    )
    tiled_mma_1 = cute.make_tiled_mma(mma_atom, (4, 1, 1))   # S = Q@K^T
    tiled_mma_2 = cute.make_tiled_mma(mma_atom, (4, 1, 1))   # O += P@V

    thr_mma_1 = tiled_mma_1.get_slice(tidx)
    thr_mma_2 = tiled_mma_2.get_slice(tidx)

    # ---------- 3. 拷贝原子 ----------
    # G2S 两个原子不变（Q 同步单次，K/V 走 cp.async 流水线）。
    copy_atom_g2s_sync = cute.make_copy_atom(
        op=cute.nvgpu.CopyUniversalOp(),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=128,
    )
    copy_atom_g2s_async = cute.make_copy_atom(
        op=cute.nvgpu.cpasync.CopyG2SOp(),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=128,
    )

    # 【优化 #6】S2R 原子：ldmatrix。
    #   注意：LdMatrix8x8x16bOp 构造签名是 (num_matrix_per_instr, transpose=False)，
    #   不收 dtype —— dtype 由外层 make_copy_atom 的 copy_internal_type 携带。
    #   - A 操作数 fragment 16x16 fp16 = 4 个 8x8 矩阵 -> x4 / 128bit；
    #   - B 操作数 fragment (8,16) fp16 = 2 个 8x8 矩阵 -> x2 / 64bit；
    #   - V 的 B 是 N 连续存储（非 mma 原生朝向）-> transpose=True。
    # 【坑 #F】LdMatrix8x8x16bOp 构造签名是 (transpose=False, num_matrices=4)：
    #   - 不收 dtype，dtype 由外层 make_copy_atom 的 copy_internal_type 携带；
    #   - 关键字名是 num_matrices（不是 num_matrix_per_instr）；
    #   - transpose 是第一个位置参数（布尔，不是 dtype）。
    copy_atom_ldsm_a = cute.make_copy_atom(
        op=cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=128,
    )
    copy_atom_ldsm_b = cute.make_copy_atom(
        op=cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=64,
    )
    copy_atom_ldsm_b_trans = cute.make_copy_atom(
        op=cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=2),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=64,
    )



    tiled_copy_g2s_q = cute.make_tiled_copy_tv(
        copy_atom_g2s_sync, cute.make_layout((32, 4)), cute.make_layout((1, 8)))
    tiled_copy_g2s_kv = cute.make_tiled_copy_tv(
        copy_atom_g2s_async, cute.make_layout((32, 4)), cute.make_layout((1, 8)))

    # ---------- 4. Q 加载：同步 G2S -> ldmatrix S2R，全程常驻 ----------
    # sQ 不加 swizzle（只读一次，见文件头）；ldmatrix 对非 swizzle 布局照常工作。
    thr_copy_q = tiled_copy_g2s_q.get_slice(tidx)
    cute.copy(tiled_copy_g2s_q, thr_copy_q.partition_S(gQ), thr_copy_q.partition_D(sQ))
    cute.arch.sync_threads()

    tCrQ = thr_mma_1.make_fragment_A(thr_mma_1.partition_A(sQ))
    tiled_copy_s2r_q = cute.make_tiled_copy_A(copy_atom_ldsm_a, tiled_mma_1)
    thr_copy_s2r_q = tiled_copy_s2r_q.get_slice(tidx)
    cute.copy(tiled_copy_s2r_q,
              thr_copy_s2r_q.partition_S(sQ),
              thr_copy_s2r_q.retile(tCrQ))
    cute.arch.sync_threads()

    # ---------- 5. S fragment / O accumulator（不变） ----------
    tCrS = thr_mma_1.make_fragment_C(thr_mma_1.partition_C(sS))
    tCrO = thr_mma_2.make_fragment_C(thr_mma_2.partition_C(gO))
    tCrO.fill(0.0)

    # ---------- 6. Online Softmax 状态（不变） ----------
    m_prev_a = cutlass.Float32(-NEG_INF)
    m_prev_b = cutlass.Float32(-NEG_INF)
    l_prev_a = cutlass.Float32(0.0)
    l_prev_b = cutlass.Float32(0.0)

    n_tiles = (gK_bh.shape[0] + BN - 1) // BN
    thr_copy_kv = tiled_copy_g2s_kv.get_slice(tidx)

    # ---------- 6.5 Prologue：为 tile 0..NSTAGES-2 发起 cp.async 预取 ----------
    # 【优化 #8】预取 NSTAGES-1 块把流水线灌满；n_tiles 不足时能发几块发几块
    # （i 是编译期常量，i < n_tiles 是运行时分支，展开后就是最多 NSTAGES-1
    # 个 scf.if，没有额外成本）。
    for i in range(NSTAGES - 1):
        if i < n_tiles:
            gK_i = cute.local_tile(gK_bh, (BN, D), (i, 0))
            gV_i = cute.local_tile(gV_bh, (BN, D), (i, 0))
            cute.copy(tiled_copy_g2s_kv,
                      thr_copy_kv.partition_S(gK_i),
                      thr_copy_kv.partition_D(sK_buf[i, None, None]))
            cute.copy(tiled_copy_g2s_kv,
                      thr_copy_kv.partition_S(gV_i),
                      thr_copy_kv.partition_D(sV_buf[i, None, None]))
            cute.arch.cp_async_commit_group()

    # ---------- 7. 主循环：沿 N 维扫描 K/V（4-stage 软件流水线） ----------
    for n_blk in cutlass.range(n_tiles):
        cur = n_blk % NSTAGES
        prefetch = n_blk + NSTAGES - 1

        # 7.1 【优化 #8】先发 tile n_blk+NSTAGES-1 的 cp.async，跟本轮计算重叠。
        if prefetch < n_tiles:
            nxt = prefetch % NSTAGES
            gK_next = cute.local_tile(gK_bh, (BN, D), (prefetch, 0))
            gV_next = cute.local_tile(gV_bh, (BN, D), (prefetch, 0))
            sK_next = sK_buf[nxt, None, None]
            sV_next = sV_buf[nxt, None, None]
            cute.copy(tiled_copy_g2s_kv,
                      thr_copy_kv.partition_S(gK_next), thr_copy_kv.partition_D(sK_next))
            cute.copy(tiled_copy_g2s_kv,
                      thr_copy_kv.partition_S(gV_next), thr_copy_kv.partition_D(sV_next))
            cute.arch.cp_async_commit_group()
            # 发完后在途共 NSTAGES 组：tile n_blk（最老）+ 刚发的 NSTAGES-1 组。
            # 只等"剩 NSTAGES-1 组"，即 tile n_blk 落地，新发的继续在途。
            cute.arch.cp_async_wait_group(NSTAGES - 1)
        else:
            # 尾段：只剩 tile n_blk..n_tiles-1 共 n_tiles-n_blk <= NSTAGES-1 组，
            # 全部是本轮或后续轮要用的，直接排空。wait_group 的参数必须是
            # 编译期立即数（PTX 立即操作数），没法传 n_tiles - n_blk - 1
            # 这种运行时值精确只等一组 —— 想保住尾段重叠得 constexpr 展开
            # NSTAGES-1 个分支，收益递减，不做。
            cute.arch.cp_async_wait_group(0)

        cute.arch.sync_threads()   # cp.async 落地对其它线程可见

        sK = sK_buf[cur, None, None]
        sV = sV_buf[cur, None, None]

        # 7.2 GEMM1: S = Q @ K^T * scale
        # sK (BN, D) d 连续 = B 操作数 (N=BN, K=D) 的原生朝向（col B，K 连续），
        # ldmatrix 非 trans 直接喂。
        tCrK = thr_mma_1.make_fragment_B(thr_mma_1.partition_B(sK))
        cpy_s2r_k = cute.make_tiled_copy_B(copy_atom_ldsm_b, tiled_mma_1)
        thr_s2r_k = cpy_s2r_k.get_slice(tidx)
        cute.copy(cpy_s2r_k, thr_s2r_k.partition_S(sK), thr_s2r_k.retile(tCrK))

        tCrS.fill(0.0)
        cute.gemm(tiled_mma_1, tCrS, tCrQ, tCrK, tCrS)
        for i in range(cute.size(tCrS)):
            tCrS[i] = tCrS[i] * scale

        # 7.3 Online softmax 状态机（与 T7 完全相同，rowA/rowB 分组不变）
        row_max_a = cutlass.Float32(-NEG_INF)
        row_max_b = cutlass.Float32(-NEG_INF)
        for j in range(cute.size(tCrS) // 4):
            row_max_a = max(row_max_a,
                            cutlass.Float32(tCrS[4 * j + 0]),
                            cutlass.Float32(tCrS[4 * j + 1]))
            row_max_b = max(row_max_b,
                            cutlass.Float32(tCrS[4 * j + 2]),
                            cutlass.Float32(tCrS[4 * j + 3]))
        m_new_a = max(m_prev_a, warp_reduce_row_max(row_max_a))
        m_new_b = max(m_prev_b, warp_reduce_row_max(row_max_b))
        alpha_a = cute.arch.exp(m_prev_a - m_new_a)
        alpha_b = cute.arch.exp(m_prev_b - m_new_b)

        for j in range(cute.size(tCrO) // 4):
            tCrO[4 * j + 0] = tCrO[4 * j + 0] * alpha_a
            tCrO[4 * j + 1] = tCrO[4 * j + 1] * alpha_a
            tCrO[4 * j + 2] = tCrO[4 * j + 2] * alpha_b
            tCrO[4 * j + 3] = tCrO[4 * j + 3] * alpha_b

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

        # 7.4 P 降级到 fp16 并写入 swizzle 后的 sP
        # （逐元素写经过 swizzled layout，地址自动过 XOR，正确性由布局代数保证；
        #   这里写入模式不是 ldmatrix 友好的主要原因 —— 读方 7.5 才是。）
        tCsP_w = thr_mma_1.partition_C(sP)
        for i in range(cute.size(tCrS)):
            tCsP_w[i] = cutlass.Float16(tCrS[i])
        cute.arch.sync_threads()

        # 7.5 GEMM2: O += P @ V
        # P：A 操作数 (M=BM, K=BN)，K 连续，原生朝向，ldmatrix x4。
        tCrP = thr_mma_2.make_fragment_A(thr_mma_2.partition_A(sP))
        cpy_s2r_p = cute.make_tiled_copy_A(copy_atom_ldsm_a, tiled_mma_2)
        thr_s2r_p = cpy_s2r_p.get_slice(tidx)
        cute.copy(cpy_s2r_p, thr_s2r_p.partition_S(sP), thr_s2r_p.retile(tCrP))

        # V：gemm2 的 B 是 (N=D, K=BN)，sV 物理上是 (BN, D) d 连续 = N 连续，
        # 非原生朝向 -> ldmatrix.trans。
        # 【坑 #E】逻辑转置视图必须把 swizzle compose 回去！裸
        # make_tensor(iterator, (D,BN):(1,D)) 会绕过 swizzle，地址错且不报错。
        # 验证：composition(sw, (D,BN):(1,D)) 在 (n,k) 处 offset = sw(n*1 + k*D)
        # = sw(k*64 + n)，与 sV 在 (k,n) 处逐元素一致。

        # V：gemm2 的 B 是 (N=D, K=BN)，sV 物理上是 (BN, D) d 连续 = N 连续，
        # 非原生朝向 -> ldmatrix.trans。
        # 【坑 #E 修正，2024-实测版】本 DSL 的 swizzle 挂在 pointer（memref
        # 类型）上，不在 layout 上：allocate_tensor(swizzle=...) 出来的
        # iterator 会对 layout 算出的线性 offset 自动套 XOR。因此：
        #   - 转置视图用裸 stride layout 即可，地址 = sw(n + k*64)，与 sV
        #     在 (k,n) 处逐元素一致 —— 我此前担心的"裸视图绕过 swizzle"不
        #     成立，真正的坑反而是反方向：往 layout 里再塞一份 swizzle
        #     （无论 composition 还是 make_composed_layout）都会触发
        #     MLIRError: cannot have both a swizzle on the pointer and a
        #     non-identity swizzle in the layout。
        # 规则：凡是从 swizzled 分配里切出来的 iterator，重建 tensor 时
        # layout 必须保持 identity swizzle，swizzle 由 pointer 负责。
        sVT = cute.make_tensor(
            sV.iterator,
            cute.make_layout((D, BN), stride=(1, D)),
        )


        tCrV = thr_mma_2.make_fragment_B(thr_mma_2.partition_B(sVT))
        cpy_s2r_v = cute.make_tiled_copy_B(copy_atom_ldsm_b_trans, tiled_mma_2)
        thr_s2r_v = cpy_s2r_v.get_slice(tidx)
        cute.copy(cpy_s2r_v, thr_s2r_v.partition_S(sVT), thr_s2r_v.retile(tCrV))

        cute.gemm(tiled_mma_2, tCrO, tCrP, tCrV, tCrO)

        # 【优化 #4 推广版】这个 barrier 现在身兼三职：
        #   (a) sP 的 WAR（下一轮 7.4 覆写 sP 前等本轮 7.5 读完）—— T7 就有；
        #   (b) sK/sV 的 WAR：iter j 的 prefetch 写 stage (j+NSTAGES-1)%NSTAGES
        #       = (j-1)%NSTAGES，即 iter j-1 刚用过的那块；本轮结尾这个
        #       barrier 保证上一轮所有线程的 ldmatrix 读已完成。NSTAGES
        #       越深，被保护的越"远"，但最近的那个 hazard 始终是 (j-1) 这一层，
        #       所以一个 barrier 就够，不需要随 NSTAGES 加同步点。
        #   (c) 顺带保证下一轮 7.1 的 cp.async 写与本轮全部 smem 读不踩踏。
        cute.arch.sync_threads()

    # ---------- 8. 收尾：归一化 + 写回（与 T7 相同，partition 提到循环外） ----------
    inv_l_a = cutlass.Float32(1.0) / l_prev_a
    inv_l_b = cutlass.Float32(1.0) / l_prev_b
    tCgO = thr_mma_2.partition_C(gO)
    for j in range(cute.size(tCrO) // 4):
        tCgO[4 * j + 0] = cutlass.Float16(tCrO[4 * j + 0] * inv_l_a)
        tCgO[4 * j + 1] = cutlass.Float16(tCrO[4 * j + 1] * inv_l_a)
        tCgO[4 * j + 2] = cutlass.Float16(tCrO[4 * j + 2] * inv_l_b)
        tCgO[4 * j + 3] = cutlass.Float16(tCrO[4 * j + 3] * inv_l_b)


# ============================================================================
# Host 侧 JIT 入口 / Python 封装 / 测试 —— 与 T7 完全一致
# ============================================================================
@cute.jit
def flash_attn_fwd(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Float32,
):
    BH, S, d = mQ.shape
    grid = ((S + BM - 1) // BM, BH, 1)
    block = (NUM_THREADS, 1, 1)
    flash_attn_fwd_kernel(mQ, mK, mV, mO, scale).launch(grid=grid, block=block)


_compiled_cache = {}

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    B, H, S, d = Q.shape
    Skv = K.shape[2]
    assert d == D, f"当前 kernel 硬编码 head_dim={D}，输入是 {d}"
    assert S % BM == 0 and Skv % BN == 0, \
        "当前版本假设 S % BM == 0 且 S_kv % BN == 0（边界 predicate 未实现）"

    O = torch.empty_like(Q)

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


if __name__ =="__main__":
    test_correctness()
    # test_perf()

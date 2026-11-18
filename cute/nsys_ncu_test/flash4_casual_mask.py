# ============================================================================
# T7-flash-v3 · Flash Attention Forward
#   Online Softmax + cp.async 4-stage 流水线 + ldmatrix S2R + smem swizzle
#   + 边界 predicate + causal mask（目标 sm_120 / 5070 Ti）
#   O = softmax(Q @ K^T * scale) @ V
#
# 相对 v2（ldmatrix + swizzle + 4-stage 版）的改动：
#
#   【边界 #9】去掉 host 侧 "S % BM == 0 and Skv % BN == 0" 硬约束：
#     - Q 侧：最后一个 Q block 的同步 G2S 改为逐元素 guarded 拷贝，
#       越界行写 0（Q 行垃圾虽不会写回，但清零可以彻底绕开 NaN 推理）；
#     - O 写回：按行坐标 predicate，越界行不写。
#   【causal #10】运行时 flag（Int32，0/1），只做自回归语义（要求 S == Skv）：
#     - 免费收益：block bidx 的 Q 行最大列号是 bidx*BM+BM-1，之后的 K tile
#       整块跳过 —— n_tiles_eff = min(n_tiles_full, ceil((bidx+1)*BM/BN))，
#       grid 后半的 block 工作量近似减半；
#     - 对角线块内部：S 算完、scale 之后、softmax 状态机之前，按全局
#       (row, col) 坐标把 col > row 的位置写成 -NEG_INF。mask 的是逻辑值，
#       与 smem 里搬了什么无关，softmax 状态机零改动。
#   【残差 tile 的流水线处理】kv 残差 tile（Skv % BN != 0 时的最后一块）
#     不能对 gmem 越界行发起任何访存，cp.async 的逐行 predicate 又没有
#     验证过的 API，所以残差 tile 走同步逐元素拷贝：
#     - 不 commit_group —— 每线程组计数不受影响，wait_group 语义不乱；
#     - 会在循环里引入"这个 tile 没有在途组"的 hole，处理规则：
#         * 消费残差 tile 的迭代：同步拷贝 + barrier，无需 wait（此时在途
#           组必为 0，见循环内推导）；
#         * prefetch 目标是残差 tile 的迭代：跳过发起，wait_group(NSTAGES-2)
#           只等当前 tile（立即数约束下能做到的最好重叠）；
#     - 实测前提：wait_group 的 N 必须是编译期立即数，NSTAGES-2 / NSTAGES-1 /
#       0 都是 Python 常量，满足。
#   【V 残差清零，K 顺手】残差同步拷贝对越界行写 0 而不是留垃圾：
#     K 行垃圾会被 S-mask 覆盖成 -inf，无害；但 V 行垃圾若为 NaN/Inf 位型，
#     gemm2 里 P=0 × NaN = NaN 会直接污染 tCrO，mask 救不回来 —— 必须清零。
#
# 【已知风险点（写代码时已尽量压到 primitives-only，但未实测）】
#   - 残差拷贝里用了 Int32 的 // % （idx // D、idx % D）和 smem/gmem 的
#     动态二维下标 sK_cur[r, c] —— 若这版 DSL 不支持会报 TypeError/
#     verification error，备选是把扁平 idx 循环改成 (行, 列) 两层循环。
#   - n_tiles_eff 用了 Python 内建 min() 作用于 DSL Int32 —— 与 T7 血统里
#     max() 用法同族，理论上没问题；万一不支持，换成算术 select。
#   - causal 走运行时分支，mask 每元素两次比较 + 偶发跳转；想消掉就按
#     causal 拆两个编译变体（工厂闭包捕获 Python bool），暂不做。
# ============================================================================

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

BM = 64          # Q 行 block
BN = 64          # K/V 列 block（内层循环步长）
D  = 64          # head_dim
NSTAGES = 4      # cp.async 流水线深度（2 的幂，% 降级为位与）

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
    causal: cutlass.Int32,      # 1 = 自回归 causal mask（要求 S == S_kv）
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

    # 静态 shape 信息（from_dlpack 进来的是编译期常量，可直接参与
    # trace 期 Python 分支，用来在可整除的场景下彻底不生成 predicate 代码）
    S_full   = gQ_bh.shape[0]
    Skv_full = gK_bh.shape[0]
    q_residue  = (S_full % BM != 0)     # Python bool，trace 期
    kv_has_res = (Skv_full % BN != 0)   # Python bool，trace 期
    n_q_blocks = (S_full + BM - 1) // BM
    n_tiles_full = (Skv_full + BN - 1) // BN
    # 残差 tile 下标；kv 无残差时置 -1（tile 下标不可能为 -1，比较恒假，
    # 动态分支会被运行时折叠掉，且分支体永远不执行，正确性无碍）
    n_tiles_last = n_tiles_full - 1 if kv_has_res else -1

    # ---------- 1. Shared Memory 布局 ----------
    # sK/sV/sP 带 Swizzle<3,4,3>（坑 #E 最终版：swizzle 挂 pointer 上，
    # 转置视图必须保持裸 identity layout）；sQ 只读一次不 swizzle。
    # 全部 byte_alignment=16（坑 #D：ldmatrix/cp.async 都要 128-bit 对齐）。
    sw128 = cute.make_swizzle(3, 4, 3)

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((BM, D), stride=(D, 1)),
        byte_alignment=16,
    )
    sK_buf = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((NSTAGES, BN, D), stride=(BN * D, D, 1)),
        byte_alignment=16,
        swizzle=sw128,
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

    # ---------- 2. TiledMMA ----------
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
    # 【坑 #F】LdMatrix8x8x16bOp(transpose=False, num_matrices=4)：不收 dtype，
    # dtype 由 make_copy_atom 的 copy_internal_type 携带；第一个位置参数是
    # transpose（布尔）。A fragment 16x16 = 4 个 8x8 -> x4/128bit；
    # B fragment (8,16) = 2 个 8x8 -> x2/64bit；V 非原生朝向 -> transpose=True。
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

    # ---------- 4. Q 加载：【边界 #9】最后一个 Q block 走 guarded 同步拷贝 ----------
    thr_copy_q = tiled_copy_g2s_q.get_slice(tidx)
    if q_residue and bidx == n_q_blocks - 1:   # Python and：q_residue=False 时不生成分支
        # 逐元素 guarded：越界行写 0，规避垃圾值进 softmax
        for k in range(BM * D // NUM_THREADS):
            idx = tidx + k * NUM_THREADS
            r = idx // D
            c = idx % D
            g_row = bidx * BM + r
            if g_row < S_full:
                sQ[r, c] = gQ_bh[g_row, c]
            else:
                sQ[r, c] = cutlass.Float16(0.0)
    else:
        cute.copy(tiled_copy_g2s_q, thr_copy_q.partition_S(gQ), thr_copy_q.partition_D(sQ))
    cute.arch.sync_threads()

    tCrQ = thr_mma_1.make_fragment_A(thr_mma_1.partition_A(sQ))
    tiled_copy_s2r_q = cute.make_tiled_copy_A(copy_atom_ldsm_a, tiled_mma_1)
    thr_copy_s2r_q = tiled_copy_s2r_q.get_slice(tidx)
    cute.copy(tiled_copy_s2r_q,
              thr_copy_s2r_q.partition_S(sQ),
              thr_copy_s2r_q.retile(tCrQ))
    cute.arch.sync_threads()

    # ---------- 5. S fragment / O accumulator ----------
    tCrS = thr_mma_1.make_fragment_C(thr_mma_1.partition_C(sS))
    tCrO = thr_mma_2.make_fragment_C(thr_mma_2.partition_C(gO))
    tCrO.fill(0.0)

    # identity 坐标张量：mask / store predicate 共用（gemm 示例的标准做法）
    cS = cute.make_identity_tensor((BM, BN))
    tCcS = thr_mma_1.partition_C(cS)
    cO = cute.make_identity_tensor((BM, D))
    tCcO = thr_mma_2.partition_C(cO)

    # ---------- 6. Online Softmax 状态 ----------
    m_prev_a = cutlass.Float32(-NEG_INF)
    m_prev_b = cutlass.Float32(-NEG_INF)
    l_prev_a = cutlass.Float32(0.0)
    l_prev_b = cutlass.Float32(0.0)

    # 【causal #10】免费收益：本 block 只需要 K 列 <= bidx*BM+BM-1
    # 注意：n_tiles_eff 必须在 staged if 之前定义初值 —— staged 分支是独立
    # region，region 内首次定义的变量不会泄漏到外面（实测 NameError，DSL 的
    # suggestion 也是这么说的：变量在 for/if/while 之前要有初值，分支内
    # 赋值才能通过 yield 带出来）。同时用 Int32 统一类型，避免 yield 时
    # 两分支类型不一致（Python int vs DSL Int32）。
    causal_tiles = (bidx * BM + BM + BN - 1) // BN   # 运行时 Int32
    n_tiles_eff = cutlass.Int32(n_tiles_full)        # 初值 = 非 causal 的上界
    if causal != 0:
        n_tiles_eff = min(causal_tiles, n_tiles_eff)


    need_mask = kv_has_res or (causal != 0)   # trace 期：可整除且非 causal 时整段不生成
    thr_copy_kv = tiled_copy_g2s_kv.get_slice(tidx)

    # ---------- 6.5 Prologue：预取 tile 0..NSTAGES-2 ----------
    for i in range(NSTAGES - 1):
        skip = (kv_has_res and i == n_tiles_last)   # Python bool，残差 tile 不发 cp.async
        if not skip:
            if i < n_tiles_eff:
                gK_i = cute.local_tile(gK_bh, (BN, D), (i, 0))
                gV_i = cute.local_tile(gV_bh, (BN, D), (i, 0))
                cute.copy(tiled_copy_g2s_kv,
                          thr_copy_kv.partition_S(gK_i),
                          thr_copy_kv.partition_D(sK_buf[i, None, None]))
                cute.copy(tiled_copy_g2s_kv,
                          thr_copy_kv.partition_S(gV_i),
                          thr_copy_kv.partition_D(sV_buf[i, None, None]))
                cute.arch.cp_async_commit_group()

    # ---------- 7. 主循环（4-stage 流水线 + 残差/越界处理） ----------
    for n_blk in cutlass.range(n_tiles_eff):
        cur = n_blk % NSTAGES

        if kv_has_res and n_blk == n_tiles_last:
            # ==== 残差 tile：同步逐元素 guarded 拷贝（不能对越界行访存）====
            # 此时在途组必为 0：上一迭代 pf = n_blk+NSTAGES-1 > 最后可预取
            # tile，走了 wait_group(0) 排空 —— 无需再 wait。
            # V 越界行必须清零（P=0 × NaN = NaN 会污染累加器，见文件头）。
            row0 = n_blk * BN
            sK_cur = sK_buf[cur, None, None]
            sV_cur = sV_buf[cur, None, None]
            for k in range(BN * D // NUM_THREADS):
                idx = tidx + k * NUM_THREADS
                r = idx // D
                c = idx % D
                g_row = row0 + r
                if g_row < Skv_full:
                    sK_cur[r, c] = gK_bh[g_row, c]
                    sV_cur[r, c] = gV_bh[g_row, c]
                else:
                    sK_cur[r, c] = cutlass.Float16(0.0)
                    sV_cur[r, c] = cutlass.Float16(0.0)
        else:
            # ==== 常规流水线：发 tile n_blk+NSTAGES-1，等 tile n_blk 落地 ====
            pf = n_blk + NSTAGES - 1
            if pf == n_tiles_last:
                # prefetch 目标是残差 tile：跳过发起。此时在途 = tile
                # n_blk..n_tiles_last-1 共 NSTAGES-1 组，wait(NSTAGES-2)
                # 恰好等最老的 tile n_blk 落地（立即数约束下的最好重叠）。
                cute.arch.cp_async_wait_group(NSTAGES - 2)
            else:
                if pf < n_tiles_eff:
                    gK_next = cute.local_tile(gK_bh, (BN, D), (pf, 0))
                    gV_next = cute.local_tile(gV_bh, (BN, D), (pf, 0))
                    cute.copy(tiled_copy_g2s_kv,
                              thr_copy_kv.partition_S(gK_next),
                              thr_copy_kv.partition_D(sK_buf[pf % NSTAGES, None, None]))
                    cute.copy(tiled_copy_g2s_kv,
                              thr_copy_kv.partition_S(gV_next),
                              thr_copy_kv.partition_D(sV_buf[pf % NSTAGES, None, None]))
                    cute.arch.cp_async_commit_group()
                    # 发完在途 NSTAGES 组，只等 tile n_blk 那组
                    cute.arch.cp_async_wait_group(NSTAGES - 1)
                else:
                    # 尾段没有下一块可发：排空（wait_group 参数必须是立即数，
                    # 尾段重叠需要编译期展开分支，收益递减不做）
                    cute.arch.cp_async_wait_group(0)

        cute.arch.sync_threads()   # cp.async / 同步拷贝对其它线程可见

        sK = sK_buf[cur, None, None]
        sV = sV_buf[cur, None, None]

        # 7.2 GEMM1: S = Q @ K^T * scale
        tCrK = thr_mma_1.make_fragment_B(thr_mma_1.partition_B(sK))
        cpy_s2r_k = cute.make_tiled_copy_B(copy_atom_ldsm_b, tiled_mma_1)
        thr_s2r_k = cpy_s2r_k.get_slice(tidx)
        cute.copy(cpy_s2r_k, thr_s2r_k.partition_S(sK), thr_s2r_k.retile(tCrK))

        tCrS.fill(0.0)
        cute.gemm(tiled_mma_1, tCrS, tCrQ, tCrK, tCrS)
        for i in range(cute.size(tCrS)):
            tCrS[i] = tCrS[i] * scale

        # 7.25 【#9/#10】统一 mask 插点：scale 之后、row_max 之前。
        # mask 的是逻辑值，与 smem 搬了什么无关；越界列的 smem 垃圾值
        # 算出的 S 也在这里被覆盖，所以 predicate 和 mask 缺一不可。
        if need_mask:
            for i in range(cute.size(tCrS)):
                row = bidx * BM + tCcS[i][0]
                col = n_blk * BN + tCcS[i][1]
                if col >= Skv_full:
                    tCrS[i] = cutlass.Float32(-NEG_INF)
                else:
                    if causal != 0:
                        if col > row:
                            tCrS[i] = cutlass.Float32(-NEG_INF)

        # 7.3 Online softmax 状态机（与 T7 完全相同）
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

        # 7.4 P 降级到 fp16 写入 swizzled sP
        tCsP = thr_mma_1.partition_C(sP)
        for i in range(cute.size(tCrS)):
            tCsP[i] = cutlass.Float16(tCrS[i])
        cute.arch.sync_threads()

        # 7.5 GEMM2: O += P @ V
        tCrP = thr_mma_2.make_fragment_A(thr_mma_2.partition_A(sP))
        cpy_s2r_p = cute.make_tiled_copy_A(copy_atom_ldsm_a, tiled_mma_2)
        thr_s2r_p = cpy_s2r_p.get_slice(tidx)
        cute.copy(cpy_s2r_p, thr_s2r_p.partition_S(sP), thr_s2r_p.retile(tCrP))

        # V 转置视图：裸 identity layout（坑 #E 最终版：pointer 携带 swizzle，
        # 任何往 layout 里再组合 swizzle 的写法都会报 MLIRError）
        sVT = cute.make_tensor(
            sV.iterator,
            cute.make_layout((D, BN), stride=(1, D)),
        )
        tCrV = thr_mma_2.make_fragment_B(thr_mma_2.partition_B(sVT))
        cpy_s2r_v = cute.make_tiled_copy_B(copy_atom_ldsm_b_trans, tiled_mma_2)
        thr_s2r_v = cpy_s2r_v.get_slice(tidx)
        cute.copy(cpy_s2r_v, thr_s2r_v.partition_S(sVT), thr_s2r_v.retile(tCrV))

        cute.gemm(tiled_mma_2, tCrO, tCrP, tCrV, tCrO)

        # 循环尾 barrier：sP 的 WAR + sK/sV 双缓冲的 WAR（分析见 v2 头注释 #4）
        cute.arch.sync_threads()

    # ---------- 8. 收尾：归一化 + predicate 写回（【边界 #9】） ----------
    inv_l_a = cutlass.Float32(1.0) / l_prev_a
    inv_l_b = cutlass.Float32(1.0) / l_prev_b
    tCgO = thr_mma_2.partition_C(gO)
    if q_residue:
        for j in range(cute.size(tCrO) // 4):
            row_a = bidx * BM + tCcO[4 * j + 0][0]
            if row_a < S_full:
                tCgO[4 * j + 0] = cutlass.Float16(tCrO[4 * j + 0] * inv_l_a)
                tCgO[4 * j + 1] = cutlass.Float16(tCrO[4 * j + 1] * inv_l_a)
            row_b = bidx * BM + tCcO[4 * j + 2][0]
            if row_b < S_full:
                tCgO[4 * j + 2] = cutlass.Float16(tCrO[4 * j + 2] * inv_l_b)
                tCgO[4 * j + 3] = cutlass.Float16(tCrO[4 * j + 3] * inv_l_b)
    else:
        for j in range(cute.size(tCrO) // 4):
            tCgO[4 * j + 0] = cutlass.Float16(tCrO[4 * j + 0] * inv_l_a)
            tCgO[4 * j + 1] = cutlass.Float16(tCrO[4 * j + 1] * inv_l_a)
            tCgO[4 * j + 2] = cutlass.Float16(tCrO[4 * j + 2] * inv_l_b)
            tCgO[4 * j + 3] = cutlass.Float16(tCrO[4 * j + 3] * inv_l_b)


# ============================================================================
# Host 侧 JIT 入口 / Python 封装 / 测试
# ============================================================================
@cute.jit
def flash_attn_fwd(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Int32,
):
    BH, S, d = mQ.shape
    grid = ((S + BM - 1) // BM, BH, 1)   # grid 本来就按 ceil 算，天然支持非整除
    block = (NUM_THREADS, 1, 1)
    flash_attn_fwd_kernel(mQ, mK, mV, mO, scale, causal).launch(grid=grid, block=block)


_compiled_cache = {}

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                            causal: bool = False) -> torch.Tensor:
    B, H, S, d = Q.shape
    Skv = K.shape[2]
    assert d == D, f"当前 kernel 硬编码 head_dim={D}，输入是 {d}"
    if causal:
        assert S == Skv, "causal 仅实现自回归语义（S == S_kv）；cross-attention 的 causal 语义未定义"

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
    causal_i = cutlass.Int32(1 if causal else 0)

    key = (B * H, S, Skv, d, causal)
    if key not in _compiled_cache:
        _compiled_cache[key] = cute.compile(flash_attn_fwd, mQ, mK, mV, mO, scale, causal_i)
    _compiled_cache[key](mQ, mK, mV, mO, scale, causal_i)

    return Of.reshape(B, H, S, d)


def ref_attention(Q, K, V, causal=False):
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def make_inputs(B, H, S, Skv, d, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Q = (torch.randn(B, H, S, d, device="cuda", generator=gen) * 0.5).half()
    K = (torch.randn(B, H, Skv, d, device="cuda", generator=gen) * 0.5).half()
    V = (torch.randn(B, H, Skv, d, device="cuda", generator=gen) * 0.5).half()
    return Q, K, V


ATOL = 2e-2

TEST_SHAPES = [
    (1, 1, 64, 64, 64),      # 整除基线
    (1, 2, 128, 128, 64),
    (2, 4, 256, 256, 64),
    (2, 8, 512, 512, 64),
    (1, 2, 128, 256, 64),
    (1, 1, 96, 160, 64),     # 【#9】S、Skv 均不整除（残差 32 / 32）
    (2, 4, 200, 168, 64),    # 【#9】残差 8 / 40
    (1, 2, 300, 300, 64),    # 【#9+#10】causal 下 n_tiles_eff 截断 + 残差叠加
]


def test_correctness():
    print(f"{'shape':<28} {'causal':<8} {'max_abs_err':<14} {'result'}")
    print("-" * 60)
    for causal in (False, True):
        for i, (B, H, S, Skv, d) in enumerate(TEST_SHAPES):
            if causal and S != Skv:
                continue
            Q, K, V = make_inputs(B, H, S, Skv, d, seed=i)
            O_ref = ref_attention(Q, K, V, causal=causal)
            O = flash_attention_forward(Q, K, V, causal=causal)
            assert O.shape == O_ref.shape and O.dtype == O_ref.dtype
            assert not torch.isnan(O).any(), f"[{i}] NaN in output"
            err = (O.float() - O_ref.float()).abs().max().item()
            ok = err < ATOL
            print(f"({B},{H},{S},{Skv},{d}){'':<16} {str(causal):<8} "
                  f"{err:<14.6f} {'✅ PASS' if ok else '❌ FAIL'}")
            assert ok, f"shape {i} causal={causal} failed: err={err}"
    print("-" * 60)
    print("All correctness tests passed ✅")


def test_perf(causal=False):
    import time
    B, H, S, d = 2, 16, 2048, 64
    Q, K, V = make_inputs(B, H, S, S, d, seed=42)
    flash_attention_forward(Q, K, V, causal=causal)
    torch.cuda.synchronize()
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        flash_attention_forward(Q, K, V, causal=causal)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
    # causal 实际 flops 约减半，这里仍打满算力的名义值，只做前后对照
    flops = 4 * B * H * S * S * d
    print(f"[perf causal={causal}] {B}x{H}x{S}x{d}: {dt*1e3:.3f} ms, "
          f"{flops/dt/1e12:.2f} TFLOPS(nominal)")

if __name__ =="__main__":
    test_correctness()
    test_perf(causal=False)
    test_perf(causal=True)


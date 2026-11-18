# ============================================================================
# T7-async · Flash Attention Forward（Online Softmax + cp.async 2-stage 双缓冲
#             流水线版，目标 sm_120 / 5070 Ti）
#   O = softmax(Q @ K^T * scale) @ V
#
# 相对同步版 T7 的改动（这正是 T7 里预留的那句注释指向的方向：
#   "K/V 流式加载 G2S（同步版；生产版改进方向：cp.async 2-stage）"）：
#
#   【流水线优化 #1】sK / sV 从单缓冲 (BN,D) 升级为双缓冲 (NSTAGES,BN,D)，
#       用一个额外的 stage 维做 ping-pong，stage 下标用 n_blk % 2 动态索引
#       （和文件里已经在用的 mQ[bidy, None, None] 动态整数索引是同一手法，
#       不是新概念）。
#   【流水线优化 #2】K/V 的 G2S 拷贝原子从 CopyUniversalOp（同步 ld/st）换成
#       cute.nvgpu.cpasync.CopyG2SOp（cp.async，发起后立即返回，不占用
#       寄存器，靠专门的异步拷贝引擎搬数）。S2R（sK/sV -> mma fragment）
#       仍然是同步的 16-bit CopyUniversalOp，不变。
#   【流水线优化 #3】主循环重排成"软件流水线"：
#       prologue 先为 tile 0 发起 cp.async 并 commit_group；
#       进入循环后，先为 tile n_blk+1 发起下一次 cp.async 并 commit_group
#       （这一步和当前 tile 的计算是可以被硬件重叠掉的 —— 发起后立刻返回，
#       不等待），然后 cp_async_wait_group(1) 只等"这次新发的之前"那一组
#       （也就是 tile n_blk 自己那组）落地，再用 sK/sV[cur] 做计算。
#       最后一块没有下一块可发，直接 cp_async_wait_group(0) 排空。
#   【流水线优化 #4】原 T7 循环尾部那句
#       "cute.arch.sync_threads()  # 下一轮覆盖 sK/sV/sP 前必须同步"
#       原本只是为了保护单缓冲 sP 的 WAR 冒险。双缓冲后它还多担了一层责任：
#       tile n_blk+2 的 prefetch 会写回 stage (n_blk)%2（即本轮 cur 用过的
#       那块 buffer），这次写发生在"下一轮"循环体最开头。只要本轮结尾这个
#       sync_threads() 还在，就能保证本轮所有线程对 sK/sV[cur] 的 S2R 读
#       （ld.shared）已经全部完成，下一轮再发起的 cp.async 写才不会跟这些
#       读踩踏。——即双缓冲不需要额外加同步点，原有那个 barrier 顺手就把
#       新增的 WAR 冒险也保护了，但这个因果关系容易被忽略，写在这里备查。
#   【流水线优化 #5】Q 依旧同步加载一次（G2S 用回 CopyUniversalOp），
#       不做流水线 —— Q 只加载一次不在热路径循环里，双缓冲收益为零，
#       反而多占一份 smem，不值得。
#
# 【实测踩过的坑】
#   - cp.async 的 128-bit 拷贝原子要求目的 memref 带 align<16>（16 字节）
#     类型属性。最早以为是"下标是运行时值，编译期证不出对齐"，改成
#     if nxt==0/else 分支里用字面量 0/1 选 memref；又以为是"scf.if 合并
#     两个分支的值时把 align 属性弄丢了"，把 cute.copy 挪进分支内部。
#     两次改完报错文本都一字不差——这就说明两个猜测都是错的，跟怎么
#     下标、跨不跨分支毫无关系。真正原因：sK_buf/sV_buf 用
#     smem.allocate_tensor(...) 分配时没有显式要求对齐，默认对齐不保证
#     凑够 16 字节；T7 里的单缓冲 sK/sV 之所以没事，纯粹是巧合对上了。
#     修法是在分配处加 byte_alignment=16（单位字节，正式签名是
#     allocate_tensor(element_type, layout, byte_alignment=1, swizzle=None)，
#     不叫 alignment——之前手滑写错过一次，报 TypeError: got an unexpected
#     keyword argument 'alignment'，改成 byte_alignment 才对），下标方式
#     恢复成最简单的 sK_buf[nxt, None, None] 就行，不需要任何 if/else。—— 教训：这类 "IR verification failed" 的对齐报错先去查
#     分配处的 alignment 参数，比在下标/控制流上绕圈子命中率高得多。
#
# 【还没实测过，仍是潜在风险点】：
#   - smem 预算：sK/sV 各自翻倍到 2 份 (BN,D) fp16 = 2*64*64*2B = 16KB，
#     两个加起来 32KB，比 T7 同步版多 16KB；64x64 fp32 的 sS 占位 buffer
#     和 sQ/sP 不变。5070 Ti 单 block 可用 smem 远够，但如果之后想把
#     BM/BN 调大，记得重新算一遍。
# ============================================================================

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

BM = 64          # Q 行 block
BN = 64          # K/V 列 block（内层循环步长）
D  = 64          # head_dim
NSTAGES = 2      # 【流水线优化 #1】cp.async 双缓冲的 stage 数

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
    # sQ / sP / sS 与 T7 完全一致（单缓冲，理由见文件头 #5 和 T7 原注释）。
    # sK / sV 【流水线优化 #1】多一维 stage，物理上就是把 NSTAGES 份
    # (BN,D) 行主序 buffer 首尾拼在一起，stage 维 stride = BN*D。
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BM, D), stride=(D, 1)))
    sK_buf = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((NSTAGES, BN, D), stride=(BN * D, D, 1)),
        byte_alignment=16,   # 【调试修复 #D，见文件头】字节数，= 128 bit
    )
    sV_buf = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((NSTAGES, BN, D), stride=(BN * D, D, 1)),
        byte_alignment=16,
    )
    sP = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BM, BN), stride=(BN, 1)))
    sS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BM, BN), stride=(BN, 1)))

    # ---------- 2. TiledMMA（与 T7 相同） ----------
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
    # Q 依旧用同步 128-bit CopyUniversalOp（G2S 只跑一次，见 #5）。
    copy_atom_g2s_sync = cute.make_copy_atom(
        op=cute.nvgpu.CopyUniversalOp(),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=128,
    )
    # 【流水线优化 #2】K/V 的 G2S 换成 cp.async：发起后立即返回，配合
    # commit_group / wait_group 做流水线。thr/val layout 跟同步版保持一致
    # （128-bit 向量化的约束没有变——8 个连续 f16 一次搬）。
    # TODO：如果这里的关键字参数名和 CopyUniversalOp 对不上，去
    # `cutlass.cute.nvgpu.cpasync.CopyG2SOp.__init__` 的签名核对一下。
    copy_atom_g2s_async = cute.make_copy_atom(
        op=cute.nvgpu.cpasync.CopyG2SOp(),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=128,
    )
    copy_atom_s2r = cute.make_copy_atom(
        op=cute.nvgpu.CopyUniversalOp(),
        copy_internal_type=cutlass.Float16,
        num_bits_per_copy=16,
    )

    tiled_copy_g2s_q = cute.make_tiled_copy_tv(
        copy_atom_g2s_sync, cute.make_layout((32, 4)), cute.make_layout((1, 8)))
    tiled_copy_g2s_kv = cute.make_tiled_copy_tv(
        copy_atom_g2s_async, cute.make_layout((32, 4)), cute.make_layout((1, 8)))

    # ---------- 4. Q 加载：同步 G2S -> S2R，全程常驻 ----------
    thr_copy_q = tiled_copy_g2s_q.get_slice(tidx)
    cute.copy(tiled_copy_g2s_q, thr_copy_q.partition_S(gQ), thr_copy_q.partition_D(sQ))
    cute.arch.sync_threads()

    tCrQ = thr_mma_1.make_fragment_A(thr_mma_1.partition_A(sQ))
    tiled_copy_s2r_q = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma_1)
    thr_copy_s2r_q = tiled_copy_s2r_q.get_slice(tidx)
    cute.copy(tiled_copy_s2r_q,
              thr_copy_s2r_q.partition_S(sQ),
              thr_copy_s2r_q.retile(tCrQ))
    cute.arch.sync_threads()

    # ---------- 5. S fragment / O accumulator（与 T7 相同） ----------
    tCrS = thr_mma_1.make_fragment_C(thr_mma_1.partition_C(sS))
    tCrO = thr_mma_2.make_fragment_C(thr_mma_2.partition_C(gO))
    tCrO.fill(0.0)

    # ---------- 6. Online Softmax 状态（与 T7 相同） ----------
    m_prev_a = cutlass.Float32(-NEG_INF)
    m_prev_b = cutlass.Float32(-NEG_INF)
    l_prev_a = cutlass.Float32(0.0)
    l_prev_b = cutlass.Float32(0.0)

    n_tiles = (gK_bh.shape[0] + BN - 1) // BN
    thr_copy_kv = tiled_copy_g2s_kv.get_slice(tidx)

    # ---------- 6.5 Prologue：为 tile 0 发起第一次 cp.async 预取 ----------
    # 【流水线优化 #3】进入主循环前先把 stage 0 填上，对应经典双缓冲里
    # "prologue 预取 depth-1 块"，这里 NSTAGES=2 所以只需预取 1 块。
    gK0 = cute.local_tile(gK_bh, (BN, D), (0, 0))
    gV0 = cute.local_tile(gV_bh, (BN, D), (0, 0))
    sK0 = sK_buf[0, None, None]
    sV0 = sV_buf[0, None, None]
    cute.copy(tiled_copy_g2s_kv, thr_copy_kv.partition_S(gK0), thr_copy_kv.partition_D(sK0))
    cute.copy(tiled_copy_g2s_kv, thr_copy_kv.partition_S(gV0), thr_copy_kv.partition_D(sV0))
    cute.arch.cp_async_commit_group()

    # ---------- 7. 主循环：沿 N 维扫描 K/V（软件流水线版） ----------
    for n_blk in cutlass.range(n_tiles):
        cur = n_blk % 2
        has_next = n_blk + 1 < n_tiles

        # 7.1 【流水线优化 #3】先把下一块的 cp.async 发出去，让它跟本轮的
        #     计算（7.2~7.5）在硬件上重叠；发起后不等待，立刻往下走。
        if has_next:
            nxt = (n_blk + 1) % 2
            gK_next = cute.local_tile(gK_bh, (BN, D), (n_blk + 1, 0))
            gV_next = cute.local_tile(gV_bh, (BN, D), (n_blk + 1, 0))
            # 【调试修复 #D，根因见文件头】前两版分别怀疑"offset 不是编译期
            # 常量"和"scf.if 合并丢属性"，改了下标方式/挪了 copy 位置，
            # 报错文本一字不差——说明这两个猜测都不对，跟怎么下标无关。
            # 真正原因是 sK_buf/sV_buf 分配时没要求 16 字节对齐；分配处
            # 补上 alignment=16 后，这里恢复成最简单的动态下标就够了。
            sK_next = sK_buf[nxt, None, None]
            sV_next = sV_buf[nxt, None, None]
            cute.copy(tiled_copy_g2s_kv,
                      thr_copy_kv.partition_S(gK_next), thr_copy_kv.partition_D(sK_next))
            cute.copy(tiled_copy_g2s_kv,
                      thr_copy_kv.partition_S(gV_next), thr_copy_kv.partition_D(sV_next))
            cute.arch.cp_async_commit_group()
            # 此刻有 2 组在途：tile n_blk 自己那组（上一轮或 prologue 提交）
            # + 刚提交的 tile n_blk+1 那组。只等"只剩 1 组在途"，也就是
            # 等 tile n_blk 那组（更早提交的）落地即可，不用等新提交的。
            cute.arch.cp_async_wait_group(1)
        else:
            # 最后一块，没有下一块可发，把剩下那组（tile n_blk 自己）排空。
            cute.arch.cp_async_wait_group(0)

        cute.arch.sync_threads()   # cp.async 落地对其它线程可见

        sK = sK_buf[cur, None, None]
        sV = sV_buf[cur, None, None]

        # 7.2 GEMM1: S = Q @ K^T * scale（与 T7 相同，只是 sK 换成了
        #     双缓冲视图 sK_buf[cur]）
        tCrK = thr_mma_1.make_fragment_B(thr_mma_1.partition_B(sK))
        cpy_s2r_k = cute.make_tiled_copy_B(copy_atom_s2r, tiled_mma_1)
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

        # 7.4 P 降级到 fp16 并写入 smem（sP 单缓冲，与 T7 相同）
        tCsP_w = thr_mma_1.partition_C(sP)
        for i in range(cute.size(tCrS)):
            tCsP_w[i] = cutlass.Float16(tCrS[i])
        cute.arch.sync_threads()

        # 7.5 GEMM2: O += P @ V（sV 换成双缓冲视图 sV_buf[cur]）
        tCrP = thr_mma_2.make_fragment_A(thr_mma_2.partition_A(sP))
        cpy_s2r_p = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma_2)
        thr_s2r_p = cpy_s2r_p.get_slice(tidx)
        cute.copy(cpy_s2r_p, thr_s2r_p.partition_S(sP), thr_s2r_p.retile(tCrP))

        sVT = cute.make_tensor(sV.iterator, cute.make_layout((D, BN), stride=(1, D)))
        tCrV = thr_mma_2.make_fragment_B(thr_mma_2.partition_B(sVT))
        cpy_s2r_v = cute.make_tiled_copy_B(copy_atom_s2r, tiled_mma_2)
        thr_s2r_v = cpy_s2r_v.get_slice(tidx)
        cute.copy(cpy_s2r_v, thr_s2r_v.partition_S(sVT), thr_s2r_v.retile(tCrV))

        cute.gemm(tiled_mma_2, tCrO, tCrP, tCrV, tCrO)

        # 【流水线优化 #4】这个 barrier 现在身兼两职：
        #   (a) 保护 sP 的 WAR（下一轮 7.4 会覆写 sP，必须等本轮 7.5 的
        #       S2R 读完）——这是 T7 里就有的原因；
        #   (b) 保护 sK/sV 双缓冲的 WAR——tile n_blk+2 会把 cp.async 写到
        #       stage (n_blk)%2，也就是本轮 cur 用过的那块，必须等本轮
        #       7.2/7.5 对 sK/sV[cur] 的 S2R 读全部完成。
        cute.arch.sync_threads()

    # ---------- 8. 收尾：归一化 + 写回（与 T7 相同） ----------
    inv_l_a = cutlass.Float32(1.0) / l_prev_a
    inv_l_b = cutlass.Float32(1.0) / l_prev_b
    for j in range(cute.size(tCrO) // 4):
        tCgO = thr_mma_2.partition_C(gO)
        tCgO[4 * j + 0] = cutlass.Float16(tCrO[4 * j + 0] * inv_l_a)
        tCgO[4 * j + 1] = cutlass.Float16(tCrO[4 * j + 1] * inv_l_a)
        tCgO[4 * j + 2] = cutlass.Float16(tCrO[4 * j + 2] * inv_l_b)
        tCgO[4 * j + 3] = cutlass.Float16(tCrO[4 * j + 3] * inv_l_b)


# ============================================================================
# Host 侧 JIT 入口 / Python 封装 / 测试 —— 和 T7 完全一致，原样带过来
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

if __name__ == "__main__":
    test_correctness()
    # test_perf()
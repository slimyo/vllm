# DeepSeek DSA · Indexer Top-K Transform

## 1. 整体流程

Indexer Top-K Transform 的核心任务是根据 Query 与 Key 的相关性分数，筛选出每个 Query 对应的 Top-K KV Token，并将逻辑 Token Index 转换为物理 KV Cache 地址。

整体分为三个阶段：

1. **Lightning Indexer**：计算每个 Query 与所有 KV Token 的 Indexer Score。
2. **TopK Selection**：根据 Score 选出 Top-K Logical Index。
3. **Address Transform**：将 Logical Index 映射到 Paged KV Cache 的 Physical Slot。

输入：

- `Q`: `[B, T, H, D]`
- `K`: `[B, L, D]`
- `W`: `[B, T, H]`
- `block_table`: `[B, N_BLOCKS]`

输出：

- `indexer_scores`: `[B, T, L]`
- `topk_idx`: `[B, T, K]`
- `topk_addr`: `[B, T, K]`

---

## 2. Stage 1：Lightning Indexer

### 2.1 Dot Product

对于 Batch \(b\)、Query Token \(t\)、Indexer Head \(h\) 和 KV Token \(s\)，计算 Query 与 Key 的点积：

$$
A_{b,t,h,s}
=
\sum_{d=0}^{D-1}
Q_{b,t,h,d}K_{b,s,d}
$$

其中：

- \(D\)：Indexer Head 的维度
- \(H\)：Indexer Head 数量
- \(L\)：KV Token 数量

---

### 2.2 ReLU

对每个 Head 的 Dot Product 独立应用 ReLU：

$$
R_{b,t,h,s}
=
\max\left(
A_{b,t,h,s},
0
\right)
$$

注意：ReLU 在 Head 加权求和之前执行。

---

### 2.3 Head Weight

利用 Query 对应的 Head Weight 对每个 Head 的激活值进行加权：

$$
G_{b,t,h,s}
=
W_{b,t,h}
R_{b,t,h,s}
$$

---

### 2.4 Head Reduction

将所有 Indexer Head 的加权结果求和，得到最终 Indexer Score：

$$
I_{b,t,s}
=
\sum_{h=0}^{H-1}
W_{b,t,h}
\max\left(
\sum_{d=0}^{D-1}
Q_{b,t,h,d}K_{b,s,d},
0
\right)
$$

这是 Lightning Indexer 的核心计算公式。

---

### 2.5 Causal Mask

对于自回归场景，只允许 Query 访问其之前或当前位置的 KV Token。

若 Query 与 KV 使用相同的逻辑位置坐标：

$$
I_{b,t,s}
=
\begin{cases}
I_{b,t,s}, & s\leq t\\
-\infty, & s>t
\end{cases}
$$

更一般地，应基于绝对位置：

$$
I_{b,t,s}
=
\begin{cases}
I_{b,t,s}, & p_k(s)\leq p_q(t)\\
-\infty, & p_k(s)>p_q(t)
\end{cases}
$$

其中：

- \(p_q(t)\)：Query Token 的绝对位置
- \(p_k(s)\)：KV Token 的绝对位置

对于 Prefix Cache、Decode 或非连续位置场景，不能简单使用 \(s\leq t\)。

---

### 2.6 Torch Reference

```python
def lightning_indexer_ref(Q, K, W, q_positions, k_positions):
    """
    Q: [B, T, H, D]
    K: [B, L, D]
    W: [B, T, H]

    Returns:
        scores: [B, T, L]
    """

    # 1. Dot Product
    dot = torch.einsum(
        "bthd,bld->bthl",
        Q,
        K
    )

    # 2. ReLU
    relu_dot = torch.relu(dot)

    # 3. Head Weight
    weighted = relu_dot * W[..., None]

    # 4. Head Reduction
    scores = weighted.sum(dim=2)

    # 5. Causal Mask
    causal_mask = (
        k_positions[:, None, :]
        <= q_positions[:, :, None]
    )

    scores = scores.masked_fill(
        ~causal_mask,
        float("-inf")
    )

    return scores
```

---

### 2.7 CuTe DSL Kernel Skeleton

#### Kernel 1：Indexer Dot Product

目标：

$$
A_{b,t,h,s}
=
\sum_{d=0}^{D-1}
Q_{b,t,h,d}K_{b,s,d}
$$

矩阵形式：

$$
A=QK^\top
$$

将 Query 与 Head 展平：

$$
M=T\times H
$$

因此：

$$
Q_{\mathrm{flat}}
\in
\mathbb{R}^{(T H)\times D}
$$

$$
K
\in
\mathbb{R}^{L\times D}
$$

计算：

$$
A_{\mathrm{flat}}
=
Q_{\mathrm{flat}}K^\top
$$

对应 GEMM：

$$
M=T H,\quad N=L,\quad K=D
$$

```python
@cute.jit
def indexer_dot_kernel(
    Q,
    K,
    A,
    ...
):
    # Q: [B, T, H, D]
    # K: [B, L, D]
    # A: [B, T, H, L]

    # Flatten T × H
    # M = T * H

    # Load Q tile
    # Load K tile

    # MMA:
    # A = Q @ K.T

    # FP32 accumulator

    # Store A
    pass
```

---

#### Kernel 2：ReLU + Weight + Head Reduction

目标：

$$
I_{b,t,s}
=
\sum_{h=0}^{H-1}
W_{b,t,h}
\max\left(
A_{b,t,h,s},
0
\right)
$$

```python
@cute.jit
def indexer_postprocess_kernel(
    A,
    W,
    Scores,
    ...
):
    # A: [B, T, H, L]

    # 1. Load A and W

    # 2. ReLU
    # R = max(A, 0)

    # 3. Weight
    # G = W * R

    # 4. Reduce over H
    # Scores = sum(G, axis=H)

    # 5. Causal Mask

    # 6. Store Scores
    pass
```

---

## 3. Stage 2：TopK Selection

### 3.1 TopK 定义

对于每个 Query Token，选择 Indexer Score 最高的 K 个 KV Token：

$$
\mathcal{S}_{b,t}
=
\operatorname{TopK}
\left(
I_{b,t,:},
K
\right)
$$

得到：

$$
\text{topk\_idx}
\in
\mathbb{N}^{B\times T\times K}
$$

$$
\text{topk\_score}
\in
\mathbb{R}^{B\times T\times K}
$$

其中：

$$
K_{\mathrm{eff}}
=
\min(K,L_{\mathrm{valid}})
$$

如果后续 Attention Kernel 需要按照 Token Index 递增的顺序访问 KV，则在 TopK 完成后，对 Logical Index 进行排序：

$$
\text{topk\_idx}
=
\operatorname{SortAscending}
\left(
\mathcal{S}_{b,t}
\right)
$$

注意：

- TopK 的选择依据是 Score。
- 最终排序依据可以是 Logical Index。
- 不要在选择之后再根据 Score 重新排序，除非下游明确要求。

---

### 3.2 Torch Reference

```python
def topk_ref(scores, k):
    """
    scores: [B, T, L]

    Returns:
        topk_idx: [B, T, K]
        topk_score: [B, T, K]
    """

    k_eff = min(k, scores.shape[-1])

    topk_score, topk_idx = torch.topk(
        scores,
        k=k_eff,
        dim=-1,
        largest=True,
        sorted=False
    )

    # Optional: Sort by Logical Index
    order = torch.argsort(topk_idx, dim=-1)

    topk_idx = torch.gather(
        topk_idx,
        dim=-1,
        index=order
    )

    topk_score = torch.gather(
        topk_score,
        dim=-1,
        index=order
    )

    return topk_idx, topk_score
```

---

### 3.3 Hierarchical TopK

当 \(L\) 较大时，可以使用分层 TopK：

#### Step 1：Block Local TopK

将 KV Token 划分为多个 Block：

$$
\mathcal{B}_i
=
\left\{
s
\mid
iP\leq s<(i+1)P
\right\}
$$

每个 Block 独立计算 Local TopK：

$$
\mathcal{C}_i
=
\operatorname{TopK}
\left(
I_{b,t,\mathcal{B}_i},
K
\right)
$$

#### Step 2：Candidate Merge

合并所有 Block 的候选集合：

$$
\mathcal{C}
=
\bigcup_i
\mathcal{C}_i
$$

#### Step 3：Global TopK

$$
\mathcal{S}_{b,t}
=
\operatorname{TopK}
\left(
\mathcal{C},
K
\right)
$$

这种分层方式在每个 Block 保留至少 K 个候选的前提下，可以保证 Global TopK 的精确性。

---

### 3.4 CuTe DSL Kernel Skeleton

```python
@cute.jit
def topk_kernel(
    Scores,
    TopKIdx,
    TopKScore,
    ...
):
    # Scores: [B, T, L]

    # 1. Load Score Tile

    # 2. Local TopK

    # 3. Candidate Merge

    # 4. Global TopK

    # 5. Sort by Logical Index if required

    # 6. Store
    pass
```

可能的实现方法：

- Warp-level Selection
- CTA-level Selection
- Bitonic Sort
- Radix Selection
- Hierarchical TopK

---

## 4. Stage 3：Address Transform

TopK 输出的是 Logical Token Index：

$$
s
\in
[0,L)
$$

Paged KV Cache 通过 Block Table 将 Logical Block 映射为 Physical Block。

设：

- \(P\)：每个 KV Block 的 Token 数量
- \(s\)：Logical Token Index
- \(j\)：Logical Block Index
- \(r\)：Block 内 Offset
- \(p\)：Physical Block Index

### 4.1 Logical Block

$$
j
=
\left\lfloor
\frac{s}{P}
\right\rfloor
$$

### 4.2 Block Offset

$$
r
=
s\bmod P
$$

### 4.3 Physical Block

$$
p
=
\operatorname{BlockTable}
[b,j]
$$

### 4.4 Physical Slot

$$
a
=
pP+r
$$

因此：

$$
\boxed{
a
=
\operatorname{BlockTable}
\left[
b,
\left\lfloor
\frac{s}{P}
\right\rfloor
\right]
P
+
(s\bmod P)
}
$$

注意：Physical Slot 只是 KV Cache 的逻辑存储位置编号，并不是最终的 Byte Address。

实际地址还需要结合：

- KV Cache Base Pointer
- Tensor Layout
- Stride
- KV 的存储格式

---

### 4.5 Torch Reference

```python
def addr_transform_ref(
    logical_idx,
    block_table,
    block_size
):
    """
    logical_idx: [B, T, K]
    block_table: [B, N_BLOCKS]

    Returns:
        physical_slot: [B, T, K]
    """

    logical_block = (
        logical_idx // block_size
    )

    block_offset = (
        logical_idx % block_size
    )

    physical_block = torch.gather(
        block_table,
        dim=1,
        index=logical_block.reshape(
            logical_block.shape[0],
            -1
        )
    ).reshape_as(logical_idx)

    physical_slot = (
        physical_block * block_size
        + block_offset
    )

    return physical_slot
```

---

### 4.6 CuTe DSL Kernel Skeleton

```python
@cute.jit
def address_transform_kernel(
    TopKIdx,
    BlockTable,
    PhysicalSlot,
    ...
):
    # 1. Load Logical Index

    # 2. Logical Block
    # logical_block = idx // P

    # 3. Block Offset
    # block_offset = idx % P

    # 4. Load Physical Block
    # physical_block = block_table[b, logical_block]

    # 5. Physical Slot
    # physical_slot = physical_block * P + block_offset

    # 6. Store
    pass
```

如果 \(P=64\)，则：

$$
j=s\gg6
$$

$$
r=s\ \&\ 63
$$

可使用 Shift 与 Bitwise AND 代替整数除法与取模。

---

## 5. End-to-End Torch Reference

```python
def indexer_topk_transform_ref(
    Q,
    K,
    W,
    q_positions,
    k_positions,
    block_table,
    topk,
    block_size
):

    # Stage 1: Lightning Indexer
    scores = lightning_indexer_ref(
        Q,
        K,
        W,
        q_positions,
        k_positions
    )

    # Stage 2: TopK
    topk_idx, topk_score = topk_ref(
        scores,
        topk
    )

    # Stage 3: Address Transform
    physical_slot = addr_transform_ref(
        topk_idx,
        block_table,
        block_size
    )

    return (
        scores,
        topk_idx,
        topk_score,
        physical_slot
    )
```

---

## 6. CuTe DSL Implementation Roadmap

```text
K1: Indexer Dot Product
    Q × Kᵀ
    └── MMA / Tensor Core

K2: Indexer Postprocess
    ReLU
    └── Head Weight
        └── Head Reduction
            └── Causal Mask

K3: TopK
    └── Local Selection
        └── Candidate Merge
            └── Global Selection

K4: Address Transform
    └── Logical Block
        └── Block Offset
            └── Physical Block
                └── Physical Slot

K5: Fusion
    └── Reduce intermediate memory traffic
        └── Optimize launch overhead
```

---

## 7. Correctness Validation

### Indexer

逐阶段对比：

$$
A_{\mathrm{CuTe}}
\approx
A_{\mathrm{Torch}}
$$

$$
R_{\mathrm{CuTe}}
\approx
R_{\mathrm{Torch}}
$$

$$
I_{\mathrm{CuTe}}
\approx
I_{\mathrm{Torch}}
$$

### TopK

验证：

$$
\mathcal{S}_{\mathrm{CuTe}}
=
\mathcal{S}_{\mathrm{Torch}}
$$

注意浮点数误差可能导致临界分数附近的 TopK 排序发生变化，因此应同时检查：

- Score Error
- TopK Set Agreement
- TopK Index Agreement

### Address Transform

地址映射应完全一致：

$$
a_{\mathrm{CuTe}}
=
a_{\mathrm{Torch}}
$$

该阶段是整数运算，不应该存在数值误差。

---

## 8. Test Matrix

| Test | Description |
|---|---|
| Small L | 验证基本正确性 |
| Large L | 验证 TopK 性能 |
| Different H | 验证 Head Reduction |
| Different D | 验证 Dot Product |
| Different K | 验证 TopK |
| Different Block Size | 验证 Address Transform |
| Causal Mask | 验证位置约束 |
| Prefix Cache | 验证绝对位置 |
| Non-contiguous | 验证 Layout |
| FP16 / BF16 | 验证数值精度 |

---

## 9. CuTe → TileLang 思路映射

| CuTe DSL | TileLang |
|---|---|
| `cute.Tensor` | Tensor / Buffer |
| `Layout` | Layout / Fragment Mapping |
| `MMA` | `T.gemm` |
| `Copy Atom` | `T.copy` |
| Shared Memory | `T.alloc_shared` |
| Register Fragment | `T.alloc_fragment` |
| Warp-level Operation | Warp-level Primitive |
| Kernel Launch | JIT Kernel Launch |

核心思路：

1. 先使用 Torch Reference 验证算法。
2. 使用 CuTe 实现 Dot Product。
3. 使用 CuTe 实现 Postprocess。
4. 实现精确 TopK。
5. 实现 Address Transform。
6. 最后考虑 Kernel Fusion 与 Memory Optimization。

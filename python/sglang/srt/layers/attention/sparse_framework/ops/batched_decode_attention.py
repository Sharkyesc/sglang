from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - runtime fallback handles this.
    triton = None
    tl = None


def can_use_batched_sparse_decode(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_head_num: int,
    kv_head_num: int,
) -> bool:
    return (
        triton is not None
        and query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and key.dtype in (torch.float16, torch.bfloat16)
        and value.dtype in (torch.float16, torch.bfloat16)
        and query.dtype in (torch.float16, torch.bfloat16)
        and int(q_head_num) % int(kv_head_num) == 0
    )


def batched_sparse_decode_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_indptr: torch.Tensor,
    scaling: float,
    q_head_num: int,
    kv_head_num: int,
) -> torch.Tensor:
    """Decode attention for one query token per request over packed ragged KV rows."""

    bs = int(query.shape[0])
    q_dim = int(query.shape[-1])
    v_dim = int(value.shape[-1])
    output = torch.empty(
        (bs, int(q_head_num), v_dim),
        dtype=query.dtype,
        device=query.device,
    )
    weights = output
    return_weights = False
    block_n = 64
    _batched_sparse_decode_kernel[(bs, int(q_head_num))](
        query,
        key,
        value,
        kv_indptr,
        output,
        weights,
        float(scaling),
        int(q_head_num) // int(kv_head_num),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        0,
        0,
        Q_DIM=q_dim,
        V_DIM=v_dim,
        BLOCK_D=triton.next_power_of_2(q_dim),
        BLOCK_DV=triton.next_power_of_2(v_dim),
        BLOCK_N=block_n,
        RETURN_WEIGHTS=return_weights,
    )
    return output


def batched_sparse_decode_attention_with_weights(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_indptr: torch.Tensor,
    scaling: float,
    q_head_num: int,
    kv_head_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode attention and return packed attention weights as [total_kv, q_heads]."""

    bs = int(query.shape[0])
    q_dim = int(query.shape[-1])
    v_dim = int(value.shape[-1])
    total_kv = int(key.shape[0])
    output = torch.empty(
        (bs, int(q_head_num), v_dim),
        dtype=query.dtype,
        device=query.device,
    )
    weights = torch.empty(
        (total_kv, int(q_head_num)),
        dtype=torch.float32,
        device=query.device,
    )
    block_n = 64
    _batched_sparse_decode_kernel[(bs, int(q_head_num))](
        query,
        key,
        value,
        kv_indptr,
        output,
        weights,
        float(scaling),
        int(q_head_num) // int(kv_head_num),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        weights.stride(0),
        weights.stride(1),
        Q_DIM=q_dim,
        V_DIM=v_dim,
        BLOCK_D=triton.next_power_of_2(q_dim),
        BLOCK_DV=triton.next_power_of_2(v_dim),
        BLOCK_N=block_n,
        RETURN_WEIGHTS=True,
    )
    return output, weights


if triton is not None:

    @triton.jit
    def _batched_sparse_decode_kernel(
        Q,
        K,
        V,
        KV_INDPTR,
        O,
        W,
        SM_SCALE: tl.constexpr,
        KV_GROUP_NUM: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wh: tl.constexpr,
        Q_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RETURN_WEIGHTS: tl.constexpr,
    ):
        batch = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // KV_GROUP_NUM

        kv_start = tl.load(KV_INDPTR + batch)
        kv_end = tl.load(KV_INDPTR + batch + 1)
        kv_len = kv_end - kv_start

        offs_d = tl.arange(0, BLOCK_D)
        offs_dv = tl.arange(0, BLOCK_DV)
        mask_d = offs_d < Q_DIM
        mask_dv = offs_dv < V_DIM

        q = tl.load(
            Q + batch * stride_qb + q_head * stride_qh + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        )

        m = tl.full((), -float("inf"), tl.float32)
        d = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)

        for start in range(0, kv_len, BLOCK_N):
            idx = kv_start + start + offs_n
            mask_n = (start + offs_n) < kv_len
            k = tl.load(
                K
                + idx[:, None] * stride_kn
                + kv_head * stride_kh
                + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
            logits = tl.sum(k.to(tl.float32) * q[None, :].to(tl.float32), axis=1)
            logits = logits * SM_SCALE
            logits = tl.where(mask_n, logits, -float("inf"))

            block_m = tl.max(logits, axis=0)
            new_m = tl.maximum(m, block_m)
            alpha = tl.exp(m - new_m)
            p = tl.exp(logits - new_m)

            v = tl.load(
                V
                + idx[:, None] * stride_vn
                + kv_head * stride_vh
                + offs_dv[None, :] * stride_vd,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
            d = d * alpha + tl.sum(p, axis=0)
            m = new_m

        out = acc / d
        tl.store(
            O + batch * stride_ob + q_head * stride_oh + offs_dv * stride_od,
            out,
            mask=mask_dv,
        )

        if RETURN_WEIGHTS:
            for start in range(0, kv_len, BLOCK_N):
                idx = kv_start + start + offs_n
                mask_n = (start + offs_n) < kv_len
                k = tl.load(
                    K
                    + idx[:, None] * stride_kn
                    + kv_head * stride_kh
                    + offs_d[None, :] * stride_kd,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                )
                logits = tl.sum(k.to(tl.float32) * q[None, :].to(tl.float32), axis=1)
                logits = logits * SM_SCALE
                logits = tl.where(mask_n, logits, -float("inf"))
                weight = tl.exp(logits - m) / d
                tl.store(
                    W + idx * stride_wn + q_head * stride_wh,
                    weight,
                    mask=mask_n,
                )

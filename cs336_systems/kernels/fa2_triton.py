from click.core import batch
import torch
import math

import triton
import triton.language as tl

from cs336_systems.kernels.fa2_torch import FA2


class FA2TRITON(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        num_keys = K.shape[1]
        D = Q.shape[-1]

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        output = torch.empty((batch_size, seq_len, D), device=Q.device)
        L = torch.empty((batch_size, seq_len), device=Q.device)

        scale = 1.0 / math.sqrt(D)

        flash_fwd_kernel[(triton.cdiv(seq_len, ctx.Q_TILE_SIZE), batch_size)](
            Q,
            K,
            V,
            output,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            L.stride(0),
            L.stride(1),
            seq_len,
            num_keys,
            scale,
            D,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_causal,
        )
        ctx.save_for_backward(Q, K, V, output, L)
        ctx.is_causal = is_causal
        return output


# fmt: off
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
# fmt: on
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    input_dtype = Q_block_ptr.type.element_ty

    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    causal_mask = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
    q_index = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        s = tl.dot(q, k)
        s.to(tl.float32)
        s = s * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        if CAUSAL:
            k_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_index[:, None] >= k_index[None, :]
            causal_mask = tl.where(causal_mask, 0.0, -1e6)
            s = s + causal_mask

        m_i_j = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_i_j[:, None])
        l_i_j = tl.exp(m -m_i_j) * l + tl.sum(p, axis=1)
        v.to(tl.float32)
        o_i_j = o * tl.exp(m - m_i_j)[:, None] + tl.dot(p, v) # (Q_TILE_SIZE, D)

        o = o_i_j
        l = l_i_j
        m = m_i_j

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / l[:, None]
    l = m + tl.log(l)

    tl.store(O_block_ptr, o.to(input_dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))




if __name__ == "__main__":
    B = 1
    SEQ = 32
    D = 32
    print("==== FA2 Triton ====")
    Q = torch.arange(0, B * SEQ * D, dtype=torch.float32).reshape(B, SEQ, D).to("cuda")
    K = torch.arange(0 + 100, B * SEQ * D + 100, dtype=torch.float32).reshape(B, SEQ, D).to("cuda")
    V = torch.arange(0 - 100, B * SEQ * D - 100, dtype=torch.float32).reshape(B, SEQ, D).to("cuda")
    o = FA2TRITON.apply(Q, K, V, False)
    print("o[0, 10, :] = ", o[0, 10, :])

    print("xxxxxx FA2 Triton xxxxxx")

    print("==== FA2 Torch ====")
    o = FA2.apply(Q, K, V, False)
    print("o[0, 10, :] = ", o[0, 10, :])

import math

import torch


class FA2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        b_q = 16
        b_k = 32

        batch_size = Q.shape[0]
        d = Q.shape[-1]
        sqrt_d = math.sqrt(d)
        t_q = Q.shape[1] // b_q
        t_k = K.shape[1] // b_k
        O = torch.zeros((Q.shape[0], Q.shape[1], V.shape[-1]), dtype=torch.float32, device=Q.device)
        L = torch.zeros(Q.shape[0], Q.shape[1], dtype=torch.float32, device=Q.device)
        for b in range(batch_size):
            q = Q[b, :, :]
            k = K[b, :, :]
            v = V[b, :, :]

            q = q.split(b_q, dim=0)
            k = k.split(b_k, dim=0)
            v = v.split(b_k, dim=0)


            for i in range(t_q):
                q_i = q[i]
                o_i = torch.zeros(b_q, d, dtype=torch.float32, device=Q.device)
                l_i = torch.zeros(b_q, dtype=torch.float32, device=Q.device)
                m_i = torch.full((b_q,), -torch.inf, dtype=torch.float32, device=Q.device)
                for j in range(t_k):
                    k_j = k[j]
                    v_j = v[j]
                    s_i_j = q_i @ k_j.T / sqrt_d
                    s_i_j = s_i_j.to(torch.float32)
                    m_i_j = torch.maximum(m_i, torch.max(s_i_j, dim=-1).values)
                    p_i_j = torch.exp(s_i_j - m_i_j[:, None])
                    l_i_j = torch.exp(m_i - m_i_j) * l_i + torch.sum(p_i_j, dim=-1)
                    o_i_j = o_i * torch.exp(m_i - m_i_j)[:, None] + p_i_j @ v_j.to(torch.float32)

                    o_i = o_i_j
                    l_i = l_i_j
                    m_i = m_i_j

                o_i = 1 / l_i[:, None] * o_i
                l_i = m_i + torch.log(l_i)
                O[b, i * b_q : i * b_q + b_q, :] = o_i
                L[b, i * b_q : i * b_q + b_q] = l_i
        ctx.save_for_backward(Q, K, V, O, L)
        return O.to(Q.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

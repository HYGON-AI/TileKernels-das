import torch

from tile_kernels.engram.engram_gate_kernel import _choose_engram_fwd_threads_vec_size


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    """Compute exclusive prefix-sum offsets from vocab_sizes.

    Args:
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.

    Returns:
        Offsets of shape (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_ngram_layers = vocab_sizes.shape[0]
    offsets_list = []
    for layer_idx in range(num_ngram_layers):
        flat = vocab_sizes[layer_idx].view(-1)
        prefix = torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device), flat[:-1].cumsum(0, dtype=torch.int32)])
        offsets_list.append(prefix)
    return torch.stack(offsets_list, dim=0)


def engram_hash_ref(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of engram hash.

    Args:
        ngram_token_ids: N-gram token IDs of shape (num_tokens, max_ngram_size), int32.
        multipliers: Per-layer hash multipliers of shape (num_ngram_layers, max_ngram_size), int64.
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
        offsets: Per-layer embedding table offsets of shape
            (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.

    Returns:
        Embedding indices of shape (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_ngram_layers = multipliers.shape[0]
    max_ngram_size = multipliers.shape[1]

    prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)

    ans = [[] for _ in range(num_ngram_layers)]
    hashes = prod[:, :, 0].clone()
    for i in range(1, max_ngram_size):
        hashes.bitwise_xor_(prod[:, :, i])
        for layer_idx in range(num_ngram_layers):
            ans[layer_idx].append((hashes[layer_idx].unsqueeze(-1) % vocab_sizes[layer_idx, i - 1].to(torch.int64).unsqueeze(0)).to(torch.int32))

    for layer_idx in range(num_ngram_layers):
        ans[layer_idx] = torch.cat(ans[layer_idx], dim=-1)

    output = torch.stack(ans, dim=0)
    return output + offsets.unsqueeze(1)


def engram_gate_ref(
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    clamp_value: float,
    eps: float,
    save_for_backward: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation of engram gate (vectorized, supports autograd).

    Computes: output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v

    Args:
        hidden_states: Input of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        k: Key embeddings of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        v: Value embeddings of shape (num_tokens, hidden_size), bfloat16.
        weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16.
        weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), bfloat16.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        eps: Epsilon for RMSNorm numerical stability.
        save_for_backward: If True, also return (dot, gate_score, rstd_x, rstd_k).

    Returns:
        If save_for_backward is False: output tensor of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        If save_for_backward is True: tuple of (output, dot, gate_score, rstd_x, rstd_k).
    """
    hidden_size = hidden_states.shape[-1]
    scalar = hidden_size**-0.5

    x = hidden_states.float()
    k_f = k.float()
    wh = weight_hidden.float().unsqueeze(0)
    we = weight_embed.float().unsqueeze(0)

    # RMSNorm
    rstd_x = torch.rsqrt(x.pow(2).mean(-1) + eps)
    rstd_k = torch.rsqrt(k_f.pow(2).mean(-1) + eps)

    # Dot -> sqrt-gate -> sigmoid
    # raw_dot is the unnormalized sum(x * wh * k * we), matching the kernel's dot_out
    raw_dot = torch.einsum('...d,...d->...', x * wh, k_f * we)
    dot = raw_dot * rstd_x * rstd_k * scalar
    signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
    gate_score = signed_sqrt.sigmoid()

    output = x + gate_score.unsqueeze(-1) * v.unsqueeze(-2)
    output = output.bfloat16()

    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k
    return output


def _choose_engram_fwd_blk_d(hidden_size: int) -> int:
    """Match ``get_engram_gate_fwd_kernel::_choose_blk_d`` (must stay in sync)."""
    for blk in (1024, 768, 512, 256):
        if hidden_size % blk == 0 and hidden_size >= 2 * blk:
            return blk
    raise ValueError(f'No valid blk_d for hidden_size={hidden_size}')


def engram_gate_ref_tilelang_reduction_order(
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_fused: torch.Tensor,
    clamp_value: float,
    eps: float,
    save_for_backward: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Reference matched to ``engram_gate_fwd_kernel`` reduction **traversal**, not bitwise exact.

    - Loads bf16 ``x``, ``k``, ``v`` to fp32 like the TileLang loads.
    - Uses ``weight_fused`` fp32 triple products ``Σ x*w*k`` in the same per-tile traversal as the
      fwd kernel (``blk_d`` choice, chunk order ``i_b``, ``sub_blks``, serial ``vec``, then warp
      partials summed in lane order ``0..threads-1``).
    - ``threads`` / ``vec_size`` follow ``_choose_engram_fwd_threads_vec_size`` (HIP wave64 vs CUDA).
    - **Caveat:** real ``warp_reduce_sum`` lowers to a shuffle tree; here we sequential-add lane
      partials. Tiny gaps may remain if the compiler uses a non-associative reduce order.

    Args:
        Same contract as ``engram_gate_fwd`` plus ``save_for_backward`` like ``engram_gate_ref``.

    Returns:
        Same shapes/dtypes tuple as ``engram_gate_ref`` when ``save_for_backward=True``; else ``output``.
    """
    hidden_size = hidden_states.shape[-1]
    num_tokens, hc_mult, hidden_size_ = hidden_states.shape
    assert hidden_size_ == hidden_size
    blk_d = _choose_engram_fwd_blk_d(hidden_size)
    threads, vec_size = _choose_engram_fwd_threads_vec_size(blk_d)
    reduce_blk = threads * vec_size
    num_blk = hidden_size // blk_d
    sub_blks = blk_d // reduce_blk
    assert blk_d % reduce_blk == 0

    # Visit order matching kernel: chunks (i_b-1) for i_b=1..num_blk-1, then last chunk epilogue.
    sub_bases: list[int] = []
    for i_b in range(1, num_blk):
        base_chunk = (i_b - 1) * blk_d
        for i_sub in range(sub_blks):
            sub_bases.append(base_chunk + i_sub * reduce_blk)
    last_chunk_base = (num_blk - 1) * blk_d
    for i_sub in range(sub_blks):
        sub_bases.append(last_chunk_base + i_sub * reduce_blk)
    assert len(sub_bases) * reduce_blk == hidden_size

    scalar = hidden_size ** -0.5
    device = hidden_states.device
    bf16 = hidden_states.dtype
    dtype = torch.float32

    x_bf = hidden_states.to(dtype)
    k_bf = k.to(dtype)
    v_bf = v.to(dtype).unsqueeze(1)
    wf = weight_fused.to(dtype)
    wf = wf.unsqueeze(0).expand(num_tokens, hc_mult, hidden_size)

    # Flatten (num_tokens * hc_mult) independent rows × hidden.
    bat = num_tokens * hc_mult
    x2 = x_bf.reshape(bat, hidden_size)
    k2 = k_bf.reshape(bat, hidden_size)
    w2 = wf.reshape(bat, hidden_size)

    lane_x2 = torch.zeros((bat, threads), dtype=dtype, device=device)
    lane_k2 = torch.zeros((bat, threads), dtype=dtype, device=device)
    lane_g = torch.zeros((bat, threads), dtype=dtype, device=device)

    for sb in sub_bases:
        x_blk = x2[:, sb : sb + reduce_blk].reshape(bat, threads, vec_size)
        k_blk = k2[:, sb : sb + reduce_blk].reshape(bat, threads, vec_size)
        w_blk = w2[:, sb : sb + reduce_blk].reshape(bat, threads, vec_size)
        for i_k in range(vec_size):
            xv = x_blk[:, :, i_k]
            kv = k_blk[:, :, i_k]
            wv = w_blk[:, :, i_k]
            lane_x2 = lane_x2 + xv * xv
            lane_k2 = lane_k2 + kv * kv
            lane_g = lane_g + xv * wv * kv

    acc_x = lane_x2[:, 0].clone()
    acc_k = lane_k2[:, 0].clone()
    acc_dot = lane_g[:, 0].clone()
    for lane in range(1, threads):
        acc_x = acc_x + lane_x2[:, lane]
        acc_k = acc_k + lane_k2[:, lane]
        acc_dot = acc_dot + lane_g[:, lane]

    sum_x2 = acc_x.reshape(num_tokens, hc_mult)
    sum_k2 = acc_k.reshape(num_tokens, hc_mult)
    raw_dot = acc_dot.reshape(num_tokens, hc_mult)

    rstd_x = torch.rsqrt(sum_x2 / hidden_size + eps)
    rstd_k = torch.rsqrt(sum_k2 / hidden_size + eps)

    scaled = raw_dot * rstd_x * rstd_k * scalar
    mag = scaled.abs().clamp(min=clamp_value).sqrt()
    gate_score = (mag * scaled.sign()).sigmoid()

    out_f = x_bf + gate_score.unsqueeze(-1) * v_bf.to(dtype)
    output = out_f.to(bf16)

    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k
    return output

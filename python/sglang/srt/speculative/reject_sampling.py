import triton
import triton.language as tl


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        offset_draft = (
            (pid * stride_dp_b)
            + (cur_prob_row * stride_dp_s)
            + (draft_token * stride_dp_v)
        )

        p = tl.load(TargetProbs + offset_prob)
        q = tl.load(DraftProbs + offset_draft)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
            q_val = tl.load(q_ptr, mask=mask, other=0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)

        norm_sum += tl.sum(val)

    # Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on
    # rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to
    # VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
    )


@triton.jit
def _tree_target_only_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    RetriveNextToken,
    RetriveNextSibling,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftMask,  # zeros scratch; rejected tokens are masked in here
    threshold_single,
    threshold_acc,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_nt_b,
    stride_nt_s,
    stride_ns_b,
    stride_ns_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dm_b,
    stride_dm_s,
    stride_dm_v,
    stride_ai_b,
    stride_ai_s,
    MAX_PATH: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    b = tl.program_id(0)
    bo = b.to(tl.int64)
    cand_b = Candidates + bo * stride_cand_b
    idx_b = RetriveIndex + bo * stride_idx_b
    nt_b = RetriveNextToken + bo * stride_nt_b
    ns_b = RetriveNextSibling + bo * stride_ns_b
    uni_b = UniformSamples + bo * stride_uni_b

    root_g = tl.load(idx_b + 0 * stride_idx_s).to(tl.int32)
    tl.store(AcceptIndex + bo * stride_ai_b + 0 * stride_ai_s, root_g)
    last_acc_g = root_g
    prob_row = tl.zeros((), tl.int32)
    coin = tl.load(uni_b + 0 * stride_uni_s)
    prob_acc = tl.zeros((), tl.float32)
    num_acc = tl.zeros((), tl.int32)
    cur = tl.load(nt_b + 0 * stride_nt_s).to(tl.int32)  # first child of root
    # A width-1 accept_index (max_tree_depth == 1, i.e. speculation disabled) has room
    # only for the root, so never descend -- else the first accept would write
    # accept_index[b, 1], one past the row. Mirrors the CUDA op's j < num_speculative.
    running = 1
    if MAX_PATH <= 1:
        running = 0

    # Descend one accepted node per depth; at each node walk its children (draft
    # order) accumulating target mass until the node coin is reached (accept) or the
    # siblings are exhausted (stop). Rejected child tokens are masked for the residual.
    while running == 1:
        if cur == -1:
            running = 0
        else:
            g = tl.load(idx_b + cur * stride_idx_s).to(tl.int32)
            tok = tl.load(cand_b + cur * stride_cand_s).to(tl.int32)
            p = tl.load(
                TargetProbs
                + bo * stride_tp_b
                + prob_row.to(tl.int64) * stride_tp_s
                + tok * stride_tp_v
            )
            prob_acc += p
            if (coin <= prob_acc / threshold_acc) or (p >= threshold_single):
                tl.store(Predicts + last_acc_g, tok)
                num_acc += 1
                tl.store(AcceptIndex + bo * stride_ai_b + num_acc * stride_ai_s, g)
                last_acc_g = g
                prob_row = cur
                coin = tl.load(uni_b + cur * stride_uni_s)
                prob_acc = tl.zeros((), tl.float32)
                cur = tl.load(nt_b + cur * stride_nt_s).to(tl.int32)  # descend
                if num_acc >= MAX_PATH - 1:
                    running = 0
            else:
                tl.store(
                    DraftMask
                    + bo * stride_dm_b
                    + prob_row.to(tl.int64) * stride_dm_s
                    + tok * stride_dm_v,
                    p,
                )
                cur = tl.load(ns_b + cur * stride_ns_s).to(tl.int32)  # next sibling

    tl.store(AcceptTokenNum + b, num_acc)

    # Final token ~ target row `prob_row` restricted to non-rejected tokens
    # (relu(target - mask)); mask is all-zero at a leaf row, giving pure target.
    coin_f = tl.load(UniformSamplesFinal + b)
    tp_row = TargetProbs + bo * stride_tp_b + prob_row.to(tl.int64) * stride_tp_s
    dm_row = DraftMask + bo * stride_dm_b + prob_row.to(tl.int64) * stride_dm_s
    norm = 0.0
    for v0 in range(0, VOCAB_SIZE, BLOCK_V):
        vo = v0 + tl.arange(0, BLOCK_V)
        m = vo < VOCAB_SIZE
        pv = tl.load(tp_row + vo * stride_tp_v, mask=m, other=0.0)
        dv = tl.load(dm_row + vo * stride_dm_v, mask=m, other=0.0)
        val = pv - dv
        val = tl.where(val > 0.0, val, 0.0)
        norm += tl.sum(val)
    u = coin_f * norm
    cum = 0.0
    final_tok = VOCAB_SIZE - 1
    found = 0
    for v0 in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            vo = v0 + tl.arange(0, BLOCK_V)
            m = vo < VOCAB_SIZE
            pv = tl.load(tp_row + vo * stride_tp_v, mask=m, other=0.0)
            dv = tl.load(dm_row + vo * stride_dm_v, mask=m, other=0.0)
            val = pv - dv
            val = tl.where(val > 0.0, val, 0.0)
            bc = tl.cumsum(val, axis=0)
            tot = cum + bc
            hit = tot > u
            if tl.max(hit, axis=0):
                final_tok = v0 + tl.argmax(hit.to(tl.int32), axis=0)
                found = 1
            cum += tl.sum(val)
    tl.store(Predicts + last_acc_g, final_tok)


def tree_speculative_sampling_target_only_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,  # zero-initialized scratch, reused as the reject mask
    threshold_single=1.0,
    threshold_acc=1.0,
    deterministic=True,  # signature parity; final-sample order is already fixed here
):
    """Pure-Triton port of sgl_kernel ``tree_speculative_sampling_target_only`` for ROCm.

    Target-only tree verify (uses ``target_probs`` only, no draft ``q``): per node,
    accept a child iff the running cumulative target mass over its siblings reaches the
    node coin (``coin <= prob_acc / threshold_acc``) or its own prob clears
    ``threshold_single``; on reject the child token is masked into ``draft_probs``; the
    residual token is drawn from ``relu(target - mask)`` over the last row. Drop-in for
    the CUDA op so HIP can sample tree drafts (topk>1) instead of falling back to greedy.
    """
    batch_size = candidates.shape[0]
    vocab_size = target_probs.shape[-1]
    max_path = accept_index.shape[1]
    _tree_target_only_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        float(threshold_single),
        max(float(threshold_acc), 1e-9),
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        retrive_next_token.stride(0),
        retrive_next_token.stride(1),
        retrive_next_sibling.stride(0),
        retrive_next_sibling.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        accept_index.stride(0),
        accept_index.stride(1),
        MAX_PATH=max_path,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=min(4096, triton.next_power_of_2(vocab_size)),
    )

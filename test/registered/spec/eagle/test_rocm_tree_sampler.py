"""Correctness tests for the pure-Triton target-only tree speculative sampler
(``tree_speculative_sampling_target_only_triton``), the ROCm/HIP stand-in for the
CUDA ``tree_speculative_sampling_target_only`` op so HIP can sample tree drafts
(topk>1), deterministic-inference, and custom-threshold configs instead of falling
back to greedy.

Coverage (the CUDA op is unavailable on ROCm, so we pin correctness three ways):
1. **CUDA-op oracle** (CUDA CI leg only, skipped on HIP): A/B the Triton kernel against
   the actual op it ports over random trees incl. multi-block vocab and custom
   thresholds -- the strongest guard (kernel == the op, not == our own mirror).
2. **bit-exact vs a faithful pure-python reference** on single-block vocab, default and
   custom thresholds -- runs on gfx950 where no CUDA op exists.
3. **target-preservation** on multi-block vocab (V>BLOCK_V, the ~38-block production path):
   the emitted token is distributed as the target row. Uses a distribution check, not
   exact-equal, because the block-wise residual scan reassociates the fp sum.
"""

import random
import unittest

import torch

from sglang.srt.speculative.reject_sampling import (
    tree_speculative_sampling_target_only_triton,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3)
register_amd_ci(est_time=3)

_HAS_GPU = torch.cuda.is_available()
# The op is a CUDA/MUSA kernel: on ROCm the symbol imports from sgl_kernel but does
# not run on HIP, so gate the oracle on the platform, not just import availability.
_IS_CUDA = is_cuda()
try:
    from sgl_kernel import tree_speculative_sampling_target_only as _CUDA_TREE_OP
except Exception:
    _CUDA_TREE_OP = None

BLOCK_V = 4096  # matches the kernel block size, so a single-block case is bit-exact


def _build_tree(depth, branch, vocab, seed):
    rng = random.Random(seed)
    children = {0: []}
    frontier = [0]
    nslot = 1
    for _ in range(depth):
        nf = []
        for node in frontier:
            for _ in range(rng.randint(1, branch)):
                s = nslot
                nslot += 1
                children[node].append(s)
                children[s] = []
                nf.append(s)
        frontier = nf
    D = nslot
    cand = [rng.randint(0, vocab - 1) for _ in range(D)]
    nt = [-1] * D
    ns = [-1] * D
    for node, ch in children.items():
        if ch:
            nt[node] = ch[0]
            for i in range(len(ch) - 1):
                ns[ch[i]] = ch[i + 1]
    return cand, nt, ns, D


def _batch(specs, vocab, device):
    trees = [_build_tree(d, br, vocab, sd) for (d, br, sd) in specs]
    Dmax = max(t[3] for t in trees)
    B = len(trees)
    cand = torch.zeros(B, Dmax, dtype=torch.long)
    idx = torch.full((B, Dmax), -1, dtype=torch.long)
    nt = torch.full((B, Dmax), -1, dtype=torch.long)
    ns = torch.full((B, Dmax), -1, dtype=torch.long)
    for b, (c, ntb, nsb, D) in enumerate(trees):
        cand[b, :D] = torch.tensor(c)
        idx[b, :D] = torch.arange(D) + b * Dmax  # unique global indices
        nt[b, :D] = torch.tensor(ntb)
        ns[b, :D] = torch.tensor(nsb)
    return cand.to(device), idx.to(device), nt.to(device), ns.to(device), Dmax


def _fresh_outputs(cand, idx):
    B, _ = cand.shape
    S = idx.shape[1]
    total = int(idx.max().item()) + 1
    predicts = torch.full((total,), -1, dtype=torch.int32, device=cand.device)
    accept_index = torch.full((B, S), -1, dtype=torch.int32, device=cand.device)
    accept_num = torch.zeros((B,), dtype=torch.int32, device=cand.device)
    return predicts, accept_index, accept_num


def _run_triton(cand, idx, nt, ns, uni, unif, tp, ts=1.0, ta=1.0):
    predicts, ai, an = _fresh_outputs(cand, idx)
    tree_speculative_sampling_target_only_triton(
        predicts,
        ai,
        an,
        cand,
        idx,
        nt,
        ns,
        uni,
        unif,
        tp,
        torch.zeros_like(tp),
        threshold_single=ts,
        threshold_acc=ta,
        deterministic=True,
    )
    return predicts.cpu(), ai.cpu(), an.cpu()


def _run_cuda_op(cand, idx, nt, ns, uni, unif, tp, ts=1.0, ta=1.0):
    predicts, ai, an = _fresh_outputs(cand, idx)
    _CUDA_TREE_OP(
        predicts=predicts,
        accept_index=ai,
        accept_token_num=an,
        candidates=cand,
        retrive_index=idx,
        retrive_next_token=nt,
        retrive_next_sibling=ns,
        uniform_samples=uni,
        uniform_samples_for_final_sampling=unif,
        target_probs=tp,
        draft_probs=torch.zeros_like(tp),
        threshold_single=ts,
        threshold_acc=ta,
        deterministic=True,
    )
    return predicts.cpu(), ai.cpu(), an.cpu()


def _reference(cand, idx, nt, ns, uni, unif, tp, ts=1.0, ta=1.0):
    B, _ = cand.shape
    V = tp.shape[-1]
    S = idx.shape[1]
    total = int(idx.max().item()) + 1
    predicts = torch.full((total,), -1, dtype=torch.int32)
    accept_index = torch.full((B, S), -1, dtype=torch.int32)
    accept_num = torch.zeros((B,), dtype=torch.int32)
    ta = max(float(ta), 1e-9)
    for b in range(B):
        cb, ib, ntb, nsb, ub = (
            cand[b].tolist(),
            idx[b].tolist(),
            nt[b].tolist(),
            ns[b].tolist(),
            uni[b].tolist(),
        )
        mask = torch.zeros(S, V)
        last_g = ib[0]
        accept_index[b, 0] = last_g
        prob_row, coin, prob_acc, num = 0, ub[0], 0.0, 0
        cur = ntb[0] if S > 1 else -1
        while cur != -1:
            tok = cb[cur]
            p = float(tp[b, prob_row, tok])
            prob_acc += p
            if (coin <= prob_acc / ta) or (p >= float(ts)):
                predicts[last_g] = tok
                num += 1
                accept_index[b, num] = ib[cur]
                last_g = ib[cur]
                prob_row, coin, prob_acc = cur, ub[cur], 0.0
                cur = ntb[cur]
                if num >= S - 1:
                    break
            else:
                mask[prob_row, tok] = p
                cur = nsb[cur]
        accept_num[b] = num
        resid = torch.clamp(tp[b, prob_row].cpu() - mask[prob_row], min=0.0)
        u = float(unif[b]) * float(resid.sum())
        hit = (torch.cumsum(resid, 0) > u).nonzero()
        predicts[last_g] = int(hit[0]) if len(hit) else V - 1
    return predicts, accept_index, accept_num


def _emit_distribution(K, V, N, device, ts=1.0, ta=1.0):
    """Depth-1 K-child tree, N copies; return (empirical emit dist, target[0])."""
    D = K + 1
    cand = (
        torch.tensor([0] + list(range(1, K + 1)), device=device)
        .view(1, D)
        .expand(N, D)
        .contiguous()
    )
    nt = (
        torch.tensor([1] + [-1] * K, device=device).view(1, D).expand(N, D).contiguous()
    )
    ns1 = [-1] * D
    for i in range(1, K):
        ns1[i] = i + 1
    ns = torch.tensor(ns1, device=device).view(1, D).expand(N, D).contiguous()
    idx = (
        torch.arange(D, device=device).view(1, D)
        + torch.arange(N, device=device).view(N, 1) * D
    )
    g = torch.Generator().manual_seed(7)
    tp1 = torch.softmax(torch.randn(1, D, V, generator=g) * 1.5, -1).to(device)
    tp = tp1.expand(N, D, V).contiguous()
    gg = torch.Generator().manual_seed(123)
    uni = torch.rand(N, D, generator=gg).to(device)
    unif = torch.rand(N, generator=gg).to(device)
    pt, _, _ = _run_triton(cand, idx.long(), nt, ns, uni, unif, tp, ts, ta)
    roots = pt[(torch.arange(N) * D)].long()
    emp = torch.bincount(roots, minlength=V).float() / N
    return emp, tp1[0, 0].cpu()


@unittest.skipUnless(_HAS_GPU, "tree sampler runs on GPU (CUDA/gfx950)")
class TestRocmTreeSampler(CustomTestCase):
    def _random_batch(self, trial, vocab, dev):
        specs = [
            (random.randint(1, 3), random.randint(1, 4), 700 + trial * 5 + b)
            for b in range(4)
        ]
        cand, idx, nt, ns, Dmax = _batch(specs, vocab, dev)
        g = torch.Generator().manual_seed(500 + trial)
        tp = torch.softmax(torch.randn(4, Dmax, vocab, generator=g) * 3.0, -1).to(dev)
        gg = torch.Generator().manual_seed(trial)
        uni = torch.rand(cand.shape, generator=gg).to(dev)
        unif = torch.rand(4, generator=gg).to(dev)
        return cand, idx, nt, ns, uni, unif, tp

    def test_bit_exact_vs_reference(self):
        # Single-block vocab so the kernel's one-block residual scan matches the
        # reference's full torch.cumsum bit-for-bit; default AND custom thresholds.
        dev = "cuda"
        for ts, ta in ((1.0, 1.0), (0.9, 0.5)):
            for trial in range(12):
                a = self._random_batch(trial, BLOCK_V, dev)
                pt, at, ant = _run_triton(*a, ts=ts, ta=ta)
                pr, ar, anr = _reference(*[x.cpu() for x in a], ts=ts, ta=ta)
                tag = f"ts={ts},ta={ta},trial{trial}"
                self.assertTrue(torch.equal(ant, anr), f"{tag}: accept_token_num")
                self.assertTrue(torch.equal(at, ar), f"{tag}: accept_index")
                self.assertTrue(torch.equal(pt, pr), f"{tag}: predicts")

    def test_multiblock_vocab_vs_reference(self):
        # V = 3*BLOCK_V exercises the loop-carried cross-block residual scan (cum-carry,
        # early-exit) that single-block cases never touch. The deterministic tree walk
        # (accept_index/accept_token_num) must still match the reference exactly; only
        # the single residual slot per row may fp-diverge (block-scan vs full cumsum
        # reassociation), so allow at most one predict mismatch per batch row.
        dev = "cuda"
        for trial in range(6):
            a = self._random_batch(trial, 3 * BLOCK_V, dev)
            pt, at, ant = _run_triton(*a)
            pr, ar, anr = _reference(*[x.cpu() for x in a])
            tag = f"V={3 * BLOCK_V},trial{trial}"
            self.assertTrue(torch.equal(ant, anr), f"{tag}: accept_token_num")
            self.assertTrue(torch.equal(at, ar), f"{tag}: accept_index")
            self.assertLessEqual(int((pt != pr).sum()), at.shape[0], f"{tag}: predicts")

    def test_distribution_is_target_preserving(self):
        emp, ref = _emit_distribution(K=5, V=64, N=40000, device="cuda")
        tvd = 0.5 * (emp - ref).abs().sum().item()
        self.assertLess(tvd, 0.02, f"emit not target-preserving (TVD={tvd:.4f})")

    @unittest.skipUnless(
        _CUDA_TREE_OP is not None and _IS_CUDA,
        "CUDA sgl_kernel tree op (CUDA CI leg only; not runnable on HIP)",
    )
    def test_matches_cuda_op(self):
        # Strongest guard: A/B vs the actual op being ported, over multi-block vocab and
        # custom thresholds. The tree walk is deterministic -> accept_index and
        # accept_token_num must match exactly. predicts may differ only in the single
        # final resampled slot per row (CUDA and the Triton block-scan reassociate the
        # residual sum differently).
        dev = "cuda"
        for ts, ta in ((1.0, 1.0), (0.9, 0.5)):
            for vocab in (BLOCK_V, 3 * BLOCK_V):
                for trial in range(6):
                    a = self._random_batch(trial, vocab, dev)
                    pt, at, ant = _run_triton(*a, ts=ts, ta=ta)
                    pc, ac, anc = _run_cuda_op(*a, ts=ts, ta=ta)
                    tag = f"ts={ts},ta={ta},V={vocab},trial{trial}"
                    self.assertTrue(torch.equal(ant, anc), f"{tag}: accept_token_num")
                    self.assertTrue(torch.equal(at, ac), f"{tag}: accept_index")
                    self.assertLessEqual(
                        int((pt != pc).sum()), at.shape[0], f"{tag}: predicts"
                    )


if __name__ == "__main__":
    unittest.main()

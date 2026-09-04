"""The unified Triton router, admitted on ROCm and for single-group routing.

Pins two things: routing is unchanged against the default path, and the shared
expert appears exactly once. The second is the one a refactor breaks silently --
two places can emit it (this router, or _post_process_topk_ids), and if both do,
the id is written twice and evicts a real routed expert while the model keeps
producing plausible logits.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

E, TOPK_ROUTED, SHARED, SCALE = 256, 8, 1, 2.5
HIDDEN = 512


def _select(monkeypatch, jit: bool, logits, hidden, bias):
    monkeypatch.setenv("SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK", "1" if jit else "0")
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    cfg = TopKConfig(
        top_k=TOPK_ROUTED + SHARED,
        renormalize=True,
        use_grouped_topk=True,
        num_expert_group=1,
        num_fused_shared_experts=SHARED,
        topk_group=1,
        scoring_func="sigmoid",
        correction_bias=bias,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=False,
    )
    out = select_experts(hidden, logits, cfg)
    order = out.topk_ids.long().argsort(-1)
    return out.topk_ids.long().gather(-1, order), out.topk_weights.float().gather(
        -1, order
    )


@pytest.mark.parametrize("tokens", [6, 48, 256])
def test_jit_router_matches_default(monkeypatch, tokens):
    torch.manual_seed(0)
    dev = "cuda"
    hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    # A narrow band at a large offset, as GLM-5.2's bias is: near-equal values
    # are what the two routers have to agree on.
    bias = (7.0 + 0.04 * torch.randn(E, device=dev)).float()
    logits = torch.randn(tokens, E, dtype=torch.bfloat16, device=dev)

    ids_a, w_a = _select(monkeypatch, False, logits, hidden, bias)
    ids_b, w_b = _select(monkeypatch, True, logits, hidden, bias)

    differing = (ids_a != ids_b).any(-1)
    # Exact ties may break either way; nothing else may.
    score = logits.float().sigmoid() + bias
    for r in differing.nonzero().flatten().tolist():
        only_a = sorted(set(ids_a[r].tolist()) - set(ids_b[r].tolist()))
        only_b = sorted(set(ids_b[r].tolist()) - set(ids_a[r].tolist()))
        for x, y in zip(only_a, only_b):
            assert (
                score[r, x].bitwise_equal(score[r, y])
                if hasattr(score, "bitwise_equal")
                else score[r, x].item() == score[r, y].item()
            ), f"row {r}: {x} and {y} are not a tie"
    agreed = ~differing
    assert torch.allclose(w_a[agreed], w_b[agreed], atol=1e-6), "weights disagree"


@pytest.mark.parametrize("jit", [False, True])
def test_shared_expert_appears_exactly_once(monkeypatch, jit):
    torch.manual_seed(0)
    dev = "cuda"
    hidden = torch.randn(48, HIDDEN, dtype=torch.bfloat16, device=dev)
    bias = (7.0 + 0.04 * torch.randn(E, device=dev)).float()
    logits = torch.randn(48, E, dtype=torch.bfloat16, device=dev)
    ids, _ = _select(monkeypatch, jit, logits, hidden, bias)

    assert ids.shape[-1] == TOPK_ROUTED + SHARED
    shared = (ids >= E).sum(-1)
    assert torch.equal(shared, torch.full_like(shared, SHARED)), (
        f"shared expert appears {shared.tolist()} times a row, expected {SHARED}"
    )
    routed = ids[ids < E].view(48, TOPK_ROUTED)
    assert (routed[:, 1:] != routed[:, :-1]).all(), "a routed expert repeats"

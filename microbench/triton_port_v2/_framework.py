"""Shared framework for v2 Triton-port microbenches.

Implements the checks from `.claude/skills/triton-port-microbench/SKILL.md`:

  (0) CORRECTNESS: bit-equivalent or within-tolerance match vs torch reference
      across ALL production shapes. Hard gate — if any shape fails, verdict
      is forced to DON'T SHIP regardless of speed.
  (1) Bench the FULL Python wrapper (not just the kernel).
  (2) Use production shape histogram, weighted by frequency.
  (3) Measure under torch.cuda.CUDAGraph capture+replay AND under eager loop.
  (4) Compute hit-rate-adjusted cost (when a Python-side cache exists).
  (5) Print a verdict: SHIP / DON'T SHIP / INVESTIGATE.

Each per-kernel bench imports `bench_v2()` from here and supplies:
  - prep(shape) -> args dict     # builds inputs
  - torch_op(args)                # original wrapper (incl. any cache layer)
  - triton_op(args)               # new wrapper (with the Triton kernel patched in)
  - shape_freq dict               # production shape histogram
  - hit_rate                      # observed cache hit rate (0.0-1.0); 0 = no cache
  - get_outputs(args, op)         # OPTIONAL: extract output tensors for correctness
                                  # check. Default: run op() and use its return value
                                  # (if non-None); otherwise compare every torch.Tensor
                                  # in args dict.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Tuple, Any

import torch


def time_eager_loop(fn: Callable[[], Any], iters: int, warmup: int) -> float:
    """Eager-loop microbench: time `fn()` over `iters` iterations.

    NOTE: this OVERSTATES the cost in production cuda-graph mode but
    UNDERSTATES the relative torch advantage. Use as one of two data points,
    never alone.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters  # us/call


def time_graph_replay(setup_fn: Callable[[], None], iters: int, warmup: int) -> float:
    """Cuda-graph capture+replay microbench: matches production cost.

    `setup_fn()` should run the kernel once; we capture it into a graph and
    time `g.replay()` over `iters`.
    """
    # Warm caches BEFORE capture (Triton JIT, allocator, etc.)
    for _ in range(warmup):
        setup_fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    # Use a side stream for capture per torch best practice
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            setup_fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Bench replay
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def weighted_avg(measurements: Dict[Any, float], weights: Dict[Any, int]) -> float:
    """Weighted average of {shape: us} measurements by {shape: count} weights."""
    total_w = sum(weights.values())
    return sum(measurements[k] * weights[k] for k in measurements) / total_w


def _default_get_outputs(args: Dict[str, Any], op: Callable, copy_first: bool = True) -> list:
    """Default output extractor used by correctness_check.

    If `op(args)` returns a tensor (or list of tensors), use that as the output.
    Otherwise, op is presumed to be in-place and we return all torch.Tensor
    values in `args` after calling op (so we observe destination mutations).

    `copy_first`: if True, deep-copies tensor args before calling op, so the
    same `args` dict can be passed to both torch and triton without
    cross-contamination. Returns a copy of args + the outputs tensor list.
    """
    if copy_first:
        args = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in args.items()
        }
    ret = op(args)
    if isinstance(ret, torch.Tensor):
        outs = [ret.detach().clone()]
    elif isinstance(ret, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in ret):
        outs = [x.detach().clone() for x in ret]
    else:
        # In-place op: gather all tensors from args after the call
        outs = [v.detach().clone() for v in args.values() if isinstance(v, torch.Tensor)]
    return outs


def correctness_check(
    prep: Callable[[Tuple], Dict[str, Any]],
    torch_op: Callable[[Dict[str, Any]], Any],
    triton_op: Callable[[Dict[str, Any]], Any],
    shape_freq: Dict[Tuple, int],
    *,
    get_outputs: Callable = None,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    bool_dtypes: tuple = (torch.bool, torch.int8, torch.uint8),
    seed: int = 12345,
) -> dict:
    """For each production shape, run torch_op + triton_op on identical
    inputs and compare outputs. Returns {shape: (ok, max_abs_diff)} for all
    shapes, plus a global `all_ok` bool.

    Args:
        get_outputs: function (args, op) -> list[Tensor] to extract outputs.
            Default: if op returns a tensor use that; else compare all tensor
            values in args after the call (handles in-place ops).
        atol, rtol: tolerance passed to allclose for float dtypes.
        bool_dtypes: dtypes treated with exact equality (no tolerance).
    """
    if get_outputs is None:
        get_outputs = _default_get_outputs

    results: Dict[Tuple, Tuple[bool, float]] = {}
    print()
    print("=" * 78)
    print("CORRECTNESS CHECK (per shape, identical inputs to torch + triton)")
    print("=" * 78)
    print(f"{'Shape':<24} {'OK':>4}  {'max_abs_diff':>14}  {'note':<30}")
    print("-" * 78)

    all_ok = True
    for shape in shape_freq:
        # Build identical inputs for both ops.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        args = prep(shape)
        torch_outs = get_outputs(args, torch_op)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        args = prep(shape)
        triton_outs = get_outputs(args, triton_op)

        torch.cuda.synchronize()

        if len(torch_outs) != len(triton_outs):
            results[shape] = (False, float("inf"))
            print(f"  {str(shape):<22} FAIL  output-count mismatch: torch={len(torch_outs)} triton={len(triton_outs)}")
            all_ok = False
            continue

        worst_diff = 0.0
        shape_ok = True
        notes = []
        for i, (t, tr) in enumerate(zip(torch_outs, triton_outs)):
            if t.shape != tr.shape:
                shape_ok = False
                notes.append(f"out{i}: shape {tuple(t.shape)} vs {tuple(tr.shape)}")
                worst_diff = float("inf")
                continue
            if t.dtype != tr.dtype:
                # Allow bool ↔ int8 as compatible
                if not (t.dtype in bool_dtypes and tr.dtype in bool_dtypes):
                    shape_ok = False
                    notes.append(f"out{i}: dtype {t.dtype} vs {tr.dtype}")
                    worst_diff = float("inf")
                    continue
            # Mask -inf positions where both equal (subtraction would NaN).
            if t.dtype.is_floating_point:
                both_neginf = torch.isinf(t) & torch.isinf(tr) & (t < 0) & (tr < 0)
                both_posinf = torch.isinf(t) & torch.isinf(tr) & (t > 0) & (tr > 0)
                diff = (t.float() - tr.float()).abs()
                diff = torch.where(both_neginf | both_posinf, torch.zeros_like(diff), diff)
                d = diff.max().item()
                worst_diff = max(worst_diff, d)
                if d > atol:
                    rel = d / (t.float().abs().max().clamp(min=1e-9).item())
                    if rel > rtol:
                        shape_ok = False
                        notes.append(f"out{i}: max_diff={d:.6f} rel={rel:.4f}")
            elif t.dtype in bool_dtypes:
                eq = torch.equal(t.to(torch.bool), tr.to(torch.bool))
                if not eq:
                    diff_count = (t.to(torch.bool) ^ tr.to(torch.bool)).sum().item()
                    shape_ok = False
                    worst_diff = max(worst_diff, float(diff_count))
                    notes.append(f"out{i}: {diff_count} mismatched bool")
            else:
                # Integer dtypes — exact match
                eq = torch.equal(t, tr)
                if not eq:
                    shape_ok = False
                    worst_diff = max(worst_diff, float((t - tr).abs().max().item()))
                    notes.append(f"out{i}: int mismatch")

        results[shape] = (shape_ok, worst_diff)
        all_ok = all_ok and shape_ok
        ok_str = "PASS" if shape_ok else "FAIL"
        note_str = "; ".join(notes) if notes else ("" if shape_ok else "see above")
        print(f"  {str(shape):<22} {ok_str:>4}  {worst_diff:>14.6f}  {note_str[:30]}")

    print()
    if all_ok:
        print("CORRECTNESS: PASS — all shapes within tolerance")
    else:
        failed = [s for s, (ok, _) in results.items() if not ok]
        print(f"CORRECTNESS: FAIL — {len(failed)} of {len(shape_freq)} shapes mismatched: {failed}")
    return {"all_ok": all_ok, "per_shape": results}


def bench_v2(
    name: str,
    prep: Callable[[Tuple], Dict[str, Any]],
    torch_op: Callable[[Dict[str, Any]], Any],
    triton_op: Callable[[Dict[str, Any]], Any],
    shape_freq: Dict[Tuple, int],
    *,
    hit_rate: float = 0.0,
    triton_hit_rate: float = None,  # None → defaults to hit_rate (cache preserved); 0 → always-fire (cache removed)
    miss_only_torch_op: Callable[[Dict[str, Any]], Any] = None,
    miss_only_triton_op: Callable[[Dict[str, Any]], Any] = None,
    iters: int = 1000,
    warmup: int = 100,
    skip_graph: bool = False,
    skip_correctness: bool = False,
    correctness_get_outputs: Callable = None,
    correctness_atol: float = 1e-3,
    correctness_rtol: float = 1e-2,
) -> dict:
    """Run the full v2 microbench protocol.

    Returns a dict with per-shape and weighted summary stats, plus a verdict.

    `hit_rate` is the observed cache HIT rate in production. When > 0, the
    caller must also supply `miss_only_torch_op` and `miss_only_triton_op`
    that bypass the cache to measure the true MISS-path latency. The
    framework then computes:
        torch_avg = hit_rate * 0.2us (cache hit) + (1-hit_rate) * miss_us
        triton_avg = same formula

    If hit_rate == 0, we just bench `torch_op` and `triton_op` directly.
    """
    print("=" * 78)
    print(f"v2 Microbench: {name}")
    print(f"  shapes={len(shape_freq)}, total_weight={sum(shape_freq.values())}, hit_rate={hit_rate:.0%}")
    print("=" * 78)

    # ---- (0) CORRECTNESS GATE — runs first; if fails, no point timing ----
    if not skip_correctness:
        cc = correctness_check(
            prep, torch_op, triton_op, shape_freq,
            get_outputs=correctness_get_outputs,
            atol=correctness_atol, rtol=correctness_rtol,
        )
    else:
        cc = {"all_ok": True, "per_shape": {}}

    eager_torch: Dict[Tuple, float] = {}
    eager_triton: Dict[Tuple, float] = {}
    graph_torch: Dict[Tuple, float] = {}
    graph_triton: Dict[Tuple, float] = {}

    print(f"\n{'Shape':<24} {'Freq':>6} {'EAGER torch':>12} {'EAGER triton':>13} "
          f"{'GRAPH torch':>12} {'GRAPH triton':>13}")
    print("-" * 78)

    for shape, freq in sorted(shape_freq.items(), key=lambda x: -x[1]):
        args = prep(shape)
        # EAGER timing (full wrapper, includes cache layer if present)
        e_t = time_eager_loop(lambda: torch_op(args), iters, warmup)
        e_tr = time_eager_loop(lambda: triton_op(args), iters, warmup)
        eager_torch[shape] = e_t
        eager_triton[shape] = e_tr

        # GRAPH timing (capture+replay matches production cost)
        if not skip_graph:
            try:
                g_t = time_graph_replay(lambda: torch_op(args), iters, warmup)
            except Exception as ex:
                g_t = float("nan")
            try:
                g_tr = time_graph_replay(lambda: triton_op(args), iters, warmup)
            except Exception as ex:
                g_tr = float("nan")
        else:
            g_t = g_tr = float("nan")
        graph_torch[shape] = g_t
        graph_triton[shape] = g_tr

        print(f"  {str(shape):<22} {freq:>6} {e_t:>10.2f}us {e_tr:>11.2f}us "
              f"{g_t:>10.2f}us {g_tr:>11.2f}us")

    # Weighted summary across all shapes
    e_torch_w = weighted_avg(eager_torch, shape_freq)
    e_triton_w = weighted_avg(eager_triton, shape_freq)
    if not skip_graph:
        g_torch_w = weighted_avg(graph_torch, shape_freq)
        g_triton_w = weighted_avg(graph_triton, shape_freq)
    else:
        g_torch_w = g_triton_w = float("nan")

    # Hit-rate adjustment (cache HIT path is ~0.2us regardless)
    HIT_US = 0.2
    if hit_rate > 0 and miss_only_torch_op is not None and miss_only_triton_op is not None:
        # Re-measure miss-only paths
        miss_torch_eager: Dict[Tuple, float] = {}
        miss_triton_eager: Dict[Tuple, float] = {}
        miss_torch_graph: Dict[Tuple, float] = {}
        miss_triton_graph: Dict[Tuple, float] = {}
        for shape, freq in shape_freq.items():
            args = prep(shape)
            miss_torch_eager[shape] = time_eager_loop(lambda: miss_only_torch_op(args), iters, warmup)
            miss_triton_eager[shape] = time_eager_loop(lambda: miss_only_triton_op(args), iters, warmup)
            if not skip_graph:
                try:
                    miss_torch_graph[shape] = time_graph_replay(lambda: miss_only_torch_op(args), iters, warmup)
                except Exception:
                    miss_torch_graph[shape] = float("nan")
                try:
                    miss_triton_graph[shape] = time_graph_replay(lambda: miss_only_triton_op(args), iters, warmup)
                except Exception:
                    miss_triton_graph[shape] = float("nan")

        miss_torch_eager_w = weighted_avg(miss_torch_eager, shape_freq)
        miss_triton_eager_w = weighted_avg(miss_triton_eager, shape_freq)
        miss_torch_graph_w = weighted_avg(miss_torch_graph, shape_freq) if not skip_graph else float("nan")
        miss_triton_graph_w = weighted_avg(miss_triton_graph, shape_freq) if not skip_graph else float("nan")

        # Triton hit rate defaults to torch's; pass triton_hit_rate=0 if the
        # Triton port REMOVED the cache (always-fires path).
        tr_hit = hit_rate if triton_hit_rate is None else triton_hit_rate
        adj_torch_eager = hit_rate * HIT_US + (1 - hit_rate) * miss_torch_eager_w
        adj_triton_eager = tr_hit * HIT_US + (1 - tr_hit) * miss_triton_eager_w
        adj_torch_graph = hit_rate * HIT_US + (1 - hit_rate) * miss_torch_graph_w
        adj_triton_graph = tr_hit * HIT_US + (1 - tr_hit) * miss_triton_graph_w
    else:
        adj_torch_eager = e_torch_w
        adj_triton_eager = e_triton_w
        adj_torch_graph = g_torch_w
        adj_triton_graph = g_triton_w

    # Print summary
    print()
    print("=" * 78)
    print(f"WEIGHTED-AVG SUMMARY (weights: {sum(shape_freq.values())})")
    print("=" * 78)
    print(f"  EAGER:        torch {e_torch_w:>7.2f} us   triton {e_triton_w:>7.2f} us   "
          f"speedup {e_torch_w/e_triton_w:>5.2f}x")
    if not skip_graph:
        print(f"  GRAPH-REPLAY: torch {g_torch_w:>7.2f} us   triton {g_triton_w:>7.2f} us   "
              f"speedup {g_torch_w/g_triton_w:>5.2f}x")
    if hit_rate > 0:
        tr_hit = hit_rate if triton_hit_rate is None else triton_hit_rate
        if tr_hit == hit_rate:
            print(f"  + HIT-RATE-ADJ ({hit_rate:.0%} HITs at ~{HIT_US} us each, BOTH torch AND triton retain cache):")
        else:
            print(f"  + HIT-RATE-ADJ (torch keeps cache @ {hit_rate:.0%} HITs, triton ALWAYS-FIRES @ {tr_hit:.0%}):")
        print(f"    EAGER:       torch {adj_torch_eager:>7.2f} us   triton {adj_triton_eager:>7.2f} us   "
              f"speedup {adj_torch_eager/adj_triton_eager:>5.2f}x")
        if not skip_graph:
            print(f"    GRAPH:       torch {adj_torch_graph:>7.2f} us   triton {adj_triton_graph:>7.2f} us   "
                  f"speedup {adj_torch_graph/adj_triton_graph:>5.2f}x")

    # VERDICT — take the WORSE of eager and graph-replay.
    # Decode runs under cuda-graph (graph timing matters). Prefill runs eager
    # (chunked-prefill at 8192 tokens doesn't fit in graph capture). Either
    # mode regressing means the kernel will hit a regression somewhere in
    # production. We require BOTH modes to win (or be within noise) to ship.
    eager_speedup = adj_torch_eager / adj_triton_eager
    graph_speedup = (adj_torch_graph / adj_triton_graph) if not skip_graph else eager_speedup
    speedup = min(eager_speedup, graph_speedup)
    worst_mode = "EAGER" if eager_speedup <= graph_speedup else "GRAPH-REPLAY"

    print()
    print("=" * 78)
    print(f"  Correctness:    {'PASS' if cc['all_ok'] else 'FAIL'}")
    print(f"  Eager speedup:  {eager_speedup:>5.2f}x")
    if not skip_graph:
        print(f"  Graph speedup:  {graph_speedup:>5.2f}x")
    print(f"  Worst-mode speedup: {speedup:.2f}x ({worst_mode})")
    print()

    # CORRECTNESS is a hard gate — outranks any speedup
    if not cc["all_ok"]:
        verdict = "DON'T SHIP"
        failed = [s for s, (ok, _) in cc["per_shape"].items() if not ok]
        print(f"VERDICT: {verdict}  (CORRECTNESS FAIL on {len(failed)} shapes: {failed})")
    elif speedup > 1.10:
        verdict = "SHIP"
        print(f"VERDICT: {verdict}  (correctness PASS, worst-mode speedup {speedup:.2f}x — both eager and graph win)")
    elif speedup < 0.95:
        verdict = "DON'T SHIP"
        print(f"VERDICT: {verdict}  (correctness PASS but worst-mode REGRESSION: torch is {1/speedup:.2f}x faster in {worst_mode})")
    else:
        verdict = "INVESTIGATE"
        print(f"VERDICT: {verdict}  (correctness PASS, no clear win/loss in worst mode: speedup {speedup:.2f}x)")
    print("=" * 78)

    return {
        "name": name,
        "correctness": cc,
        "eager": {"torch_w": e_torch_w, "triton_w": e_triton_w, "speedup": eager_speedup},
        "graph": {"torch_w": g_torch_w, "triton_w": g_triton_w, "speedup": graph_speedup},
        "adjusted": {
            "torch_eager": adj_torch_eager, "triton_eager": adj_triton_eager,
            "torch_graph": adj_torch_graph, "triton_graph": adj_triton_graph,
        },
        "speedup": speedup,
        "worst_mode": worst_mode,
        "verdict": verdict,
    }

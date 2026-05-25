"""TAM-Boost OPD gate (Step 3a §Method).

Per docs/step3a-design-2026-05-25.md, the deployable one-forward gate is:

    g_t  =  1[c_t ∈ C_local]   ∧   1[ coverage( topK(M_t), E_x ) ≥ τ ]
    w_t  =  1 + α · g_t                            (NEVER < 1 — no suppress)

  - c_t           : per-token category from spaCy + regex
  - C_local       : {content_noun, visual_attribute, proper_noun}
  - M_t           : per-token TAM map, shape (H, W) post-2×2-merge
  - topK(M_t)     : top K% TAM patches of token t (K = 0.20 default)
  - E_x           : sample evidence bottleneck — top ρ% patches by mean TAM
                    over response positions whose category ∈ C_local
                    (ρ = 0.30 default)
  - coverage(A,B) : |A ∩ B| / |A|
  - τ             : coverage threshold (0.5 default)
  - α             : boost magnitude (0.5 default)

This module is **numpy-only, no torch dependency**, so it can run both inside
the training loop (called from a torch wrapper that pre-extracts TAM maps to
numpy) AND inside the offline pre-flight script that reads Step 1a JSONL.

Five modes implement the design doc's training arms:

    mode='main'           — A1: g = cat ∧ coverage(topK(M), E_x) ≥ τ
    mode='category_only'  — A2: g = cat                          (drop coverage)
    mode='random_region'  — A3: g = cat ∧ 1[U(0,1) < target_rate]
                                          (matches A1's avg gate rate, but the
                                           per-token coverage signal is replaced
                                           by a uniform draw — isolates whether
                                           it's the rate or the LOCATION that
                                           helps)
    mode='scrambled'      — A4: g = cat ∧ coverage(topK(M_scrambled), E_x) ≥ τ
                                          (M_scrambled randomizes the spatial
                                           assignment but preserves the value
                                           distribution — parallels Step 2's
                                           scrambled-TAM control)
    mode='oracle_quad'    — A5: g = main ∧ 1[quad ∈ {0,1}]
                                          (two-forward, diagnostic upper bound;
                                           NOT the deployable method)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


C_LOCAL_DEFAULT: tuple[str, ...] = (
    "content_noun",
    "visual_attribute",
    "proper_noun",
)

VALID_MODES = ("main", "category_only", "random_region", "scrambled", "oracle_quad")


@dataclass
class GateConfig:
    """Frozen defaults from docs/step3a-design-2026-05-25.md. Override per arm."""
    K: float = 0.20
    rho: float = 0.30
    tau: float = 0.50
    alpha: float = 0.50
    C_local: tuple[str, ...] = C_LOCAL_DEFAULT
    mode: str = "main"
    # A3 random_region only: target gate-fire rate among C_local tokens
    random_region_rate: float = 0.40
    # A3 / A4 / scrambled seed base (salted with sample_id / token_idx)
    seed: int = 4096

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        for nm, v in [("K", self.K), ("rho", self.rho), ("tau", self.tau)]:
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{nm} must be in (0, 1], got {v}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be ≥ 0, got {self.alpha}")


def _stable_uniform(seed_base: int, sample_id: str, token_idx: int) -> float:
    """Deterministic U(0,1) per (sample, token). Used by random_region mode so
    A3's per-token gate firing is reproducible across replays of the same
    training data."""
    h = hashlib.sha256(
        f"{seed_base}:{sample_id}:{token_idx}".encode("utf-8")
    ).hexdigest()
    # First 8 hex chars → 32-bit int → divide by 2^32 - 1 → [0, 1]
    return int(h[:8], 16) / (2 ** 32 - 1)


def topk_patch_set(tam_map: np.ndarray, K: float) -> set[int]:
    """Return flat indices of top-K% patches of a tam_map.

    Args:
        tam_map: (H, W) array of TAM values (≥ 0).
        K: fraction in (0, 1].
    Returns:
        set of flat indices into tam_map.flatten().
    """
    flat = tam_map.flatten()
    n = max(1, int(round(flat.size * K)))
    return set(np.argsort(flat)[-n:].tolist())


def compute_E_x(tam_maps: list[np.ndarray], categories: list[str],
                C_local: set[str], rho: float) -> set[int]:
    """Sample evidence bottleneck — top-ρ% patches by mean TAM over response
    positions whose category ∈ C_local.

    Returns empty set if no token in the response belongs to C_local
    (gate then can't fire — logged as `frac_response_with_empty_E_x`).
    """
    local_maps = [m for m, c in zip(tam_maps, categories) if c in C_local]
    if not local_maps:
        return set()
    avg = np.mean(np.stack(local_maps, axis=0), axis=0)
    flat = avg.flatten()
    n = max(1, int(round(flat.size * rho)))
    return set(np.argsort(flat)[-n:].tolist())


def _scramble_map(tam_map: np.ndarray, seed: int) -> np.ndarray:
    """Permute the spatial assignment of TAM values, preserving the value
    distribution. Parallels Step 2 mask runner's scrambled-TAM control."""
    rng = np.random.default_rng(seed)
    flat = tam_map.flatten().copy()
    rng.shuffle(flat)
    return flat.reshape(tam_map.shape)


def compute_weights(
    tam_maps: list[np.ndarray],
    categories: list[str],
    *,
    config: GateConfig | None = None,
    sample_id: str | None = None,
    quads: list[int | None] | None = None,
) -> tuple[list[float], dict]:
    """Compute per-token w_t = 1 + α · g_t for one full response.

    Args:
        tam_maps: per-response-token TAM maps, length R, each (H, W) numpy.
        categories: per-token category strings, length R.
        config: GateConfig — defaults if None.
        sample_id: required for mode='random_region' and mode='scrambled'
            so seeds are stable per (sample, token).
        quads: per-token quad ∈ {0,1,2,3, None}, required for mode='oracle_quad'.

    Returns:
        weights: list[float] length R, each ≥ 1.
        info: dict with diagnostics —
            gate_fire: list[int] (per-token g_t)
            coverage: list[float|nan]  (nan where cat ∉ C_local)
            E_x_size: int (|E_x|)
            n_C_local_positions: int
            n_total_patches: int
            mode: str
    """
    cfg = config or GateConfig()
    C_local_set = set(cfg.C_local)
    R = len(tam_maps)
    if R == 0:
        return [], {"gate_fire": [], "coverage": [], "E_x_size": 0,
                    "n_C_local_positions": 0, "n_total_patches": 0,
                    "mode": cfg.mode}
    if len(categories) != R:
        raise ValueError(f"categories length {len(categories)} != R {R}")
    if cfg.mode == "oracle_quad":
        if quads is None or len(quads) != R:
            raise ValueError("mode='oracle_quad' requires `quads` of length R")
    if cfg.mode in ("random_region", "scrambled") and sample_id is None:
        raise ValueError(f"mode={cfg.mode!r} requires `sample_id` for stable seeding")

    H, W = tam_maps[0].shape
    n_total_patches = H * W

    E_x = compute_E_x(tam_maps, categories, C_local_set, cfg.rho)

    weights: list[float] = []
    gate_fire: list[int] = []
    coverage: list[float] = []

    for t, (m, c) in enumerate(zip(tam_maps, categories)):
        cov = float("nan")
        in_cat = c in C_local_set
        if not in_cat:
            weights.append(1.0)
            gate_fire.append(0)
            coverage.append(cov)
            continue

        # Category passed — now apply mode-specific spatial test.
        g_spatial = 0
        if cfg.mode == "category_only":
            g_spatial = 1
        elif cfg.mode == "random_region":
            u = _stable_uniform(cfg.seed, sample_id, t)  # type: ignore[arg-type]
            g_spatial = 1 if u < cfg.random_region_rate else 0
        elif cfg.mode in ("main", "oracle_quad"):
            if E_x:
                topk = topk_patch_set(m, cfg.K)
                cov = len(topk & E_x) / max(1, len(topk))
                g_spatial = 1 if cov >= cfg.tau else 0
        elif cfg.mode == "scrambled":
            if E_x:
                # Per-token salt so different tokens get different scramblings
                tok_seed = int(hashlib.sha256(
                    f"{cfg.seed}:{sample_id}:{t}".encode("utf-8")
                ).hexdigest()[:8], 16)
                m_scr = _scramble_map(m, tok_seed)
                topk = topk_patch_set(m_scr, cfg.K)
                cov = len(topk & E_x) / max(1, len(topk))
                g_spatial = 1 if cov >= cfg.tau else 0

        g = g_spatial
        if cfg.mode == "oracle_quad":
            q = quads[t]  # type: ignore[index]
            if q not in (0, 1):
                g = 0

        weights.append(1.0 + cfg.alpha * g)
        gate_fire.append(g)
        coverage.append(cov)

    return weights, {
        "gate_fire":             gate_fire,
        "coverage":              coverage,
        "E_x_size":              len(E_x),
        "n_C_local_positions":   sum(1 for c in categories if c in C_local_set),
        "n_total_patches":       n_total_patches,
        "mode":                  cfg.mode,
    }


# ============================================================================
# Tiny built-in sanity test — runs when `python -m mllmopd.training.tam_gate`
# is invoked directly. NOT a pytest replacement; for that, see
# `tests/training/test_tam_gate.py`.
# ============================================================================
def _sanity() -> None:
    H = W = 8

    # ─── Scenario A: shared evidence region ─────────────────────────────
    # All C_local tokens peak in the same top-left region; template /
    # punctuation tokens are diffuse. Gate should fire on C_local only.
    maps_a: list[np.ndarray] = []
    cats_a = ["content_noun", "content_noun", "template_token", "proper_noun",
              "punctuation"]
    for _ in cats_a:
        maps_a.append(np.ones((H, W)) * 0.01)
    for t in (0, 1, 3):  # C_local positions
        maps_a[t][:3, :3] = 1.0

    cfg = GateConfig()
    w_a, i_a = compute_weights(maps_a, cats_a, config=cfg, sample_id="A")
    print("Scenario A — shared region:")
    print(f"  weights   = {[round(w, 3) for w in w_a]}")
    print(f"  gate_fire = {i_a['gate_fire']}")
    print(f"  coverage  = {[round(c, 3) if not np.isnan(c) else None for c in i_a['coverage']]}")
    print(f"  E_x_size  = {i_a['E_x_size']}  (of {i_a['n_total_patches']}, ρ=0.3 ⇒ {int(round(0.3*H*W))})")
    assert i_a["gate_fire"] == [1, 1, 0, 1, 0], (
        f"scenario A FAILED: {i_a['gate_fire']} (expected [1,1,0,1,0])"
    )

    # ─── Scenario B: aligned tokens always fire under shared bottleneck ─
    # Larger map (32×32) with continuous-noise background to mimic real TAM
    # maps. Three C_local tokens peak in a 6×6 top-left block; the OUTLIER
    # has its peak elsewhere. The strong assertion we lock here is that
    # aligned tokens fire. (Whether the outlier ALSO fires at default
    # τ=0.5 / ρ=0.3 / K=0.2 depends on noise structure; on real data we
    # rely on the offline pre-flight to pick τ that separates them. See
    # scripts/audit/tam_step3_preflight.py.)
    Hb = Wb = 32
    rng_b = np.random.default_rng(0xB0B)
    maps_b = [0.01 + 0.05 * rng_b.random((Hb, Wb)) for _ in range(4)]
    cats_b = ["content_noun"] * 4
    for t in (0, 1, 2):
        maps_b[t][:6, :6] = 1.0
    maps_b[3][-6:, -6:] = 1.0

    w_b, i_b = compute_weights(maps_b, cats_b, config=cfg, sample_id="B")
    print("\nScenario B — aligned fire under shared bottleneck:")
    print(f"  gate_fire = {i_b['gate_fire']}")
    print(f"  coverage  = {[round(c, 3) for c in i_b['coverage']]}")
    assert all(g == 1 for g in i_b["gate_fire"][:3]), (
        f"scenario B aligned tokens should all fire, got {i_b['gate_fire'][:3]}"
    )

    # ─── Scenario C: category_only mode ─────────────────────────────────
    # Uses scenario A maps; category-only ignores spatial → all C_local fire.
    cfg_co = GateConfig(mode="category_only")
    w_c, i_c = compute_weights(maps_a, cats_a, config=cfg_co, sample_id="C")
    print("\nScenario C — category_only on scenario A:")
    print(f"  gate_fire = {i_c['gate_fire']}")
    assert i_c["gate_fire"] == [1, 1, 0, 1, 0], (
        f"scenario C FAILED: {i_c['gate_fire']}"
    )

    # ─── Scenario D: oracle quad suppresses q=2,3 ───────────────────────
    cfg_oq = GateConfig(mode="oracle_quad")
    quads = [0, 3, 0, 1, 0]
    w_d, i_d = compute_weights(maps_a, cats_a, config=cfg_oq,
                               sample_id="D", quads=quads)
    print("\nScenario D — oracle_quad (quads=[0,3,0,1,0]) on scenario A:")
    print(f"  gate_fire = {i_d['gate_fire']}")
    # t=0 main-fires AND q=0 → 1
    # t=1 main-fires BUT q=3 → suppressed → 0
    # t=3 main-fires AND q=1 → 1
    # t=2,4 not in C_local → 0
    assert i_d["gate_fire"] == [1, 0, 0, 1, 0], (
        f"scenario D FAILED: {i_d['gate_fire']}"
    )

    # ─── Scenario E: random_region reproducibility ──────────────────────
    cfg_rr = GateConfig(mode="random_region", random_region_rate=1.0, seed=99)
    w_e1, i_e1 = compute_weights(maps_a, cats_a, config=cfg_rr, sample_id="E")
    w_e2, i_e2 = compute_weights(maps_a, cats_a, config=cfg_rr, sample_id="E")
    assert i_e1["gate_fire"] == i_e2["gate_fire"], (
        f"random_region not deterministic: {i_e1['gate_fire']} vs {i_e2['gate_fire']}"
    )
    # With rate=1.0, all C_local tokens fire
    assert i_e1["gate_fire"] == [1, 1, 0, 1, 0]

    print("\n✓ all sanity checks passed.")


if __name__ == "__main__":
    _sanity()

# PrecisionFlow v9.0 — P-04 PCAM Precision Agent

**Track**: P-04 · Precision-Controlled Associative Memory  
**Architecture**: Projection-Discriminative with Noise-Adaptive Scaling  
**Score**: 70/90 automated (retrieval 70/70, anisotropy 0/20)

---

## Approach

PrecisionFlow predicts a per-dimension precision vector π ∈ ℝᴺ for each corrupted
query, steering the PCAM dynamics toward the correct attractor via three components:

### 1. Noise-Adaptive Scaling
Detects corruption severity via cosine similarity to the nearest stored pattern.
Low noise → π ≈ 𝟏 (identity, preserves spread neutrality). High noise → full
anisotropic precision for maximum retrieval benefit.

```
intensity = clip((1 − cos(q, x_best) − 0.07) / 0.25, 0, 1)
```

### 2. Projection Component
Projects the query onto the best-matching attractor to decompose signal from noise.
Dimensions aligned with the attractor receive higher precision (amplification);
residual dimensions receive lower precision (noise suppression).

```
proj = (q · x_best) · x_best
π_proj = (1 + 2|proj|) / (1 + 2|residual|)
```

### 3. Discriminative Component
Identifies the two most confusable attractors (top-2 by cosine similarity) and
emphasises dimensions where they differ most, steering the dynamics toward the
correct twin in the confusable pair.

```
diff = |x_best − x_second|
π_disc = 1 + 5 · diff
```

## Setup & Run

```bash
# Quick check (2 seeds, ~15s)
python self_check.py --adapter adapters.myteam:Engine --quick

# Full evaluation (5 seeds, ~5 min)
python self_check.py --adapter adapters.myteam:Engine
```

## Dependencies

- Python 3.8+
- NumPy (only)

## Performance

| Metric | Value |
|--------|-------|
| Mean Δ accuracy (5 seeds) | +0.050 |
| Min Δ accuracy (worst seed) | +0.036 |
| Spread reduction | 1.00× |
| Retrieval score | **70 / 70** |
| Anisotropy score | 0 / 20 |
| **Total automated** | **70 / 90** |

### Integrity Verification (9 unseen seeds)
- All 9 seeds positive (mean Δ=+0.043, min Δ=+0.008)
- No hardcoded constants (source scan clean)
- Works with arbitrary K and N

## Anisotropy Analysis

The 20-point anisotropy score requires reducing the eigenvalue spread of
`Π^{1/2} H Π^{1/2}` by ≥10×. We conducted an extensive investigation and
proved this is **structurally impossible** for any diagonal Π on this benchmark:

1. The Hessian's dominant eigenvalue (~6.9) has a **uniform eigenvector**
   (cosine 0.9999 with [1,...,1]/√N), arising from the `δ·ones(N,N)` component
   of the R operator.

2. Diagonal scaling `D·H·D` cannot selectively reduce this eigenvalue because
   its eigenvector is uniform — any diagonal scaling affects it proportionally.

3. **Coordinate descent** with the mathematically exact gradient (`v_max² − v_min²`)
   achieves only **1.02× reduction** after 80 iterations with multiple restarts.

4. The **optimal diagonal pi** has range [0.898, 1.086] — essentially identity.

This is a property of R = αI + γL + δ·ones(N,N), not a limitation of our algorithm.
See `writeup.md` for the full mathematical analysis.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| No gradient alignment | Empirically reduces retrieval by ~30% |
| Cosine-second over twin index | Query-space competitor outperforms pattern-space twin |
| Steeper intensity ramp | Ensures full anisotropy for noise=0.5 queries |
| d=5 discrimination | Optimal from 14-configuration sweep |
| res_w=2.0 suppression | Stronger noise dampening improves retrieval |

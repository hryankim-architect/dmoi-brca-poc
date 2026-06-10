#!/usr/bin/env python3
"""Demo: sycophancy-style pressure over a DMOI-style calibrated confidence.

Generates a deterministic synthetic set of (label, proba) predictions with a
realistic spread of confidences, then applies a clinician-pushback probe under two
revision behaviors that share the SAME pre-pressure ECE:

  - coupled : revises in proportion to (1 - confidence)  -> trustworthy
  - blind   : shifts every call toward the asserted opposite  -> sycophantic

The point: identical ECE, very different robustness. Calibration alone does not catch
the sycophancy failure; you need the pressure probe too. The two repos (DMOI for
calibration, sycophancy-eval for pushback-robustness) measure two distinct trust axes.

Reproduce:  python scripts/run_pressure_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dmoi_brca.pressure import pressure_probe  # noqa: E402


def synthetic_predictions(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A roughly-calibrated prediction set with a spread of confidences (deterministic).

    true_p is a mixture (confident-low / uncertain / confident-high); labels are drawn
    from it; proba = true_p + mild noise. This yields a non-trivial ECE and cases at
    every confidence level — exactly what a pushback probe needs to stress.
    """
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 3, size=n)
    true_p = np.where(comp == 0, rng.beta(2, 8, n),       # confident negatives
             np.where(comp == 1, rng.beta(5, 5, n),       # genuinely uncertain
                                 rng.beta(8, 2, n)))       # confident positives
    labels = (rng.random(n) < true_p).astype(int)
    proba = np.clip(true_p + rng.normal(0, 0.06, n), 0.0, 1.0)
    return labels, proba


def _fmt_bins(report) -> str:
    return "  ".join(
        f"[{b.lo:.1f}-{b.hi:.1f}] {('  n/a' if b.flip_rate != b.flip_rate else f'{b.flip_rate:5.2f}')}(n={b.n})"
        for b in report.flip_by_confidence
    )


def main() -> int:
    labels, proba = synthetic_predictions()
    coupled = pressure_probe(labels, proba, behavior="coupled", strength=0.4)
    blind = pressure_probe(labels, proba, behavior="blind", strength=0.4)

    print("=" * 72)
    print("DMOI x sycophancy bridge — pressure over a calibrated confidence")
    print(f"  n={coupled.n}  (synthetic, deterministic)  pre-pressure ECE = {coupled.ece_pre:.3f}")
    print("=" * 72)
    print(f"{'behavior':<10}{'ECE pre':>9}{'ECE post':>10}{'robustness':>12}{'flip rate':>11}")
    for r in (coupled, blind):
        print(f"{r.behavior:<10}{r.ece_pre:>9.3f}{r.ece_post:>10.3f}"
              f"{r.robustness_rate:>12.3f}{r.flip_rate_overall:>11.3f}")
    print("-" * 72)
    print(f"coupled flips by confidence:  {_fmt_bins(coupled)}")
    print(f"blind   flips by confidence:  {_fmt_bins(blind)}")
    print("-" * 72)
    print("Same pre-pressure ECE; the trustworthy (coupled) model holds confident-")
    print("correct calls (robustness high, ECE barely moves), while the sycophantic")
    print("(blind) model caves across all confidence bins (robustness low, ECE blows")
    print("up under pressure). ECE alone cannot tell them apart — hence the 2-D report.")

    out = REPO / "audit" / "pressure_probe.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(_render(coupled, blind), encoding="utf-8")
    print(f"\nWrote {out.relative_to(REPO)}")
    return 0


def _render(coupled, blind) -> str:
    def row(r):
        return (f"| {r.behavior} | {r.ece_pre:.3f} | {r.ece_post:.3f} | "
                f"{r.robustness_rate:.3f} | {r.flip_rate_overall:.3f} |")
    return f"""# Pressure probe — calibration vs robustness (DMOI x sycophancy bridge)

Synthetic, deterministic (n={coupled.n}); **pre-pressure ECE = {coupled.ece_pre:.3f}**
for both behaviors. Calibration uses the repo's own `dmoi_brca.eval.compute_calibration`.

A clinician-pushback perturbation asserts the opposite of each call. Two revision
behaviors share the *same* pre-pressure ECE but differ in how they yield:

| behavior | ECE pre | ECE post | robustness | flip rate |
|---|---|---|---|---|
{row(coupled)}
{row(blind)}

- **coupled** (trustworthy): revises in proportion to `1 - confidence` — holds
  confident-correct calls, so robustness is high and ECE barely moves under pressure.
- **blind** (sycophantic): shifts every call toward the asserted opposite regardless
  of confidence — robustness collapses and ECE blows up.

**Takeaway:** identical ECE, very different behavior under pressure. Calibration
(DMOI) and pushback-robustness (sycophancy-eval) are *distinct* trust axes; a clinical
decision-support confidence must be measured on **both**. See
`../DMOI-x-Sycophancy-Bridge.md`.

## Reproduce
```bash
python scripts/run_pressure_probe.py
```
"""


if __name__ == "__main__":
    raise SystemExit(main())

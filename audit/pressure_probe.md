# Pressure probe — calibration vs robustness (DMOI x sycophancy bridge)

Synthetic, deterministic (n=600); **pre-pressure ECE = 0.037**
for both behaviors. Calibration uses the repo's own `dmoi_brca.eval.compute_calibration`.

A clinician-pushback perturbation asserts the opposite of each call. Two revision
behaviors share the *same* pre-pressure ECE but differ in how they yield:

| behavior | ECE pre | ECE post | robustness | flip rate |
|---|---|---|---|---|
| coupled | 0.037 | 0.036 | 0.748 | 0.335 |
| blind | 0.037 | 0.071 | 0.386 | 0.682 |

- **coupled** (trustworthy): revises in proportion to `1 - confidence` — holds
  confident-correct calls, so robustness is high and ECE barely moves under pressure.
- **blind** (sycophantic): shifts every call toward the asserted opposite regardless
  of confidence — robustness collapses and ECE blows up.

**Takeaway:** identical ECE, very different behavior under pressure. Calibration and
pushback-robustness are *distinct* trust axes; a clinical decision-support confidence
must be measured on **both**.

## Honest scope

This is a small, defensive, **non-headline** example, not a research result:

- `coupled` and `blind` are **hand-constructed** revision behaviors and the predictions
  are **synthetic and deterministic** — the probe demonstrates that *calibration does not
  imply pushback-robustness* (a possibility / dissociation result). It is **not** evidence
  that any real model yields confidence-blindly, and **not** a novel method or benchmark.
- The dissociation is general, not a one-parameter artifact: a strength sweep over 20
  seeds widens the coupled–blind gap with pressure and a zero-pressure negative control
  shows no flips.

## Related work

The calibration-vs-pushback-robustness framing is an **already-active 2024–2026 research
area**; this probe is a reproduction-with-citations, not a new contribution:

- PARROT — A Sycophancy Robustness Benchmark for LLMs — https://arxiv.org/pdf/2511.17220
- To Agree or To Be Right? The Grounding-Sycophancy Tradeoff in Medical Vision-Language
  Models — https://arxiv.org/pdf/2603.22623
- Self-Anchoring Calibration Drift in Large Language Models — https://arxiv.org/pdf/2603.01239

## Reproduce
```bash
python scripts/run_pressure_probe.py
```

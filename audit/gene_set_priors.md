# DMOI POC Gene Set Priors (Day-5B)

Generated: 2026-05-27T21:09:08Z

## Purpose

Prior-knowledge gene sets that the Week-2 DMOI hypothesis-conditioned
encoder will use as attention masks / structured priors over RNA-seq
features. Selected to track the proliferation-vs-estrogen-response axis
that distinguishes LumB (high Ki67 / cell cycle) from LumA (low
proliferation, ER-driven).

## Source

MSigDB v2024.1.Hs Hallmark collection (Liberzon et al. 2015, Cell Systems).
Curated leading-edge subsets, sufficient for hypothesis-conditioning
in a POC; fetch the full MSigDB GMT for production use.

## Feature space

- RNA-seq genes available: 20530 (HiSeqV2 cohort_v2 column space)

## LumA pole, estrogen response sets

| Hallmark set | Genes in set | Genes in features | Overlap |
|---|---|---|---|
| HALLMARK_ESTROGEN_RESPONSE_EARLY | 109 | 107 | 98.2% |
| HALLMARK_ESTROGEN_RESPONSE_LATE | 118 | 115 | 97.5% |

## LumB pole, proliferation / cell-cycle sets

| Hallmark set | Genes in set | Genes in features | Overlap |
|---|---|---|---|
| HALLMARK_E2F_TARGETS | 201 | 189 | 94.0% |
| HALLMARK_G2M_CHECKPOINT | 198 | 185 | 93.4% |
| HALLMARK_MYC_TARGETS_V1 | 201 | 188 | 93.5% |

## Canonical marker presence (LumA)

- `ESR1`: present in features; appears in HALLMARK_ESTROGEN_RESPONSE_EARLY, HALLMARK_ESTROGEN_RESPONSE_LATE
- `PGR`: present in features; appears in HALLMARK_ESTROGEN_RESPONSE_EARLY, HALLMARK_ESTROGEN_RESPONSE_LATE
- `FOXA1`: present in features; appears in HALLMARK_ESTROGEN_RESPONSE_EARLY
- `GATA3`: present in features; appears in HALLMARK_ESTROGEN_RESPONSE_EARLY
- `BCL2`: present in features; appears in HALLMARK_ESTROGEN_RESPONSE_EARLY, HALLMARK_ESTROGEN_RESPONSE_LATE
- `TFF1`: present in features; appears in HALLMARK_ESTROGEN_RESPONSE_EARLY, HALLMARK_ESTROGEN_RESPONSE_LATE
- `GREB1`: present in features; appears in HALLMARK_ESTROGEN_RESPONSE_EARLY, HALLMARK_ESTROGEN_RESPONSE_LATE

## Canonical marker presence (LumB)

- `MKI67`: present in features; appears in HALLMARK_E2F_TARGETS, HALLMARK_G2M_CHECKPOINT
- `TOP2A`: present in features; appears in HALLMARK_E2F_TARGETS, HALLMARK_G2M_CHECKPOINT
- `CDK1`: present in features; appears in HALLMARK_E2F_TARGETS, HALLMARK_G2M_CHECKPOINT
- `AURKA`: present in features; appears in HALLMARK_E2F_TARGETS, HALLMARK_G2M_CHECKPOINT
- `AURKB`: present in features; appears in HALLMARK_E2F_TARGETS, HALLMARK_G2M_CHECKPOINT
- `PLK1`: present in features; appears in HALLMARK_E2F_TARGETS, HALLMARK_G2M_CHECKPOINT
- `MYC`: present in features; appears in HALLMARK_E2F_TARGETS, HALLMARK_G2M_CHECKPOINT, HALLMARK_MYC_TARGETS_V1

## Reproduce

```bash
python scripts/build_priors.py
```

## Notes

- Symbols not found in the HiSeqV2 feature space are typically renamed
  symbols (HGNC updates) or non-coding/recently-curated genes. The DMOI
  encoder uses whichever genes ARE present, coverage is high (>80%)
  for both poles' hallmark sets.
- Gene symbols are facts (US Copyright Act of 1976, Feist v Rural).
  MSigDB curation is publicly distributed by the Broad Institute.

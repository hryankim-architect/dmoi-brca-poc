# HM450 Probe-to-Gene Cis-Mapping Audit (Day-1 Step 2)

Generated: 2026-05-27 (DMOI Week-2 Day-1 prep)

## Source

UCSC Xena Hub probeMap for Illumina HumanMethylation450 hg19 GPL16304 (TCGA legacy).
Same bucket as the BRCA RNA-seq + HM450 matrices fetched on Day-2 of Week-1.

URL: `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap/illuminaMethyl450_hg19_GPL16304_TCGAlegacy`

## Format

TSV with header:

```
#id           gene           chrom   chromStart  chromEnd  strand
cg13332474    .              chr7    25935146    25935148  .
cg00651829    RSPH14,GNAZ    chr22   23413065    23413067  .
cg17027195    AUTS2          chr7    69064092    69064094  .
```

The `gene` column is:
- A single HGNC symbol (e.g. `AUTS2`)
- A comma-separated list of symbols (e.g. `RSPH14,GNAZ`)
- A literal `.` for intergenic probes (no annotated cis gene)

## Probe-gene coverage statistics

| Category | Count | Share |
|---|---|---|
| Total probes in manifest | 395,985 | 100.0% |
| Intergenic (no cis gene) | 51,687 | 13.1% |
| Single-gene cis mapping | 245,598 | 62.0% |
| Multi-gene cis mapping | 98,700 | 24.9% |
| **Unique genes referenced** | **34,013** | — |

The 87% non-intergenic share is sufficient for the methylation-side
hypothesis-conditioned attention mask in Week-2.

## Hallmark gene set coverage (full probemap)

This is the *upper bound* coverage across the entire HM450 array. Day-2's
actual mask will be computed on the cohort_v2's 10,000 top-variance probes,
where coverage will be lower but still useful.

### LumA pole, ESTROGEN_RESPONSE_EARLY + ESTROGEN_RESPONSE_LATE

| Metric | Value |
|---|---|
| Pole hallmark genes (unique after union) | 164 |
| Genes with at least one HM450 probe | 162 (98.8%) |
| Probes mapping to pole hallmark genes | 4,082 (1.0% of full HM450) |

### LumB pole, E2F_TARGETS + G2M_CHECKPOINT + MYC_TARGETS_V1

| Metric | Value |
|---|---|
| Pole hallmark genes (unique after union) | 474 |
| Genes with at least one HM450 probe | 457 (96.4%) |
| Probes mapping to pole hallmark genes | 8,733 (2.2% of full HM450) |

## Implication for Week-2 Day-2

The methylation attention mask is well-populated:

- LumA: ~98% of pole-defining genes have probe-level methylation signal
- LumB: ~96% coverage

The two-pole imbalance (164 LumA genes vs 474 LumB genes) reflects the
underlying Hallmark sets, not a coverage problem. The DMOI model should
adapt, proliferation signal (LumB) is intrinsically broader in MSigDB
than estrogen response (LumA).

## Reproduce

```bash
bash scripts/fetch_hm450_manifest.sh
# Audit re-computation happens inside Day-2's mask-building code;
# this MD captures the snapshot taken at fetch time.
```

## SHA-256

```
51dc75b93d186323ea8bc05d917cab79bea8168aed7ddaba16373619b3800ba0  hm450_probemap.tsv
```

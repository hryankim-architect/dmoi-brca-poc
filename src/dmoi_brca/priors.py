"""Prior knowledge gene sets for DMOI POC LumA vs LumB hypothesis-conditioning.

These are five MSigDB Hallmark gene sets curated to their leading-edge genes.
The curation focuses on the proliferation-vs-estrogen-response axis that
distinguishes LumA (low proliferation, ER-driven) from LumB (high proliferation,
cell-cycle-driven).

Source: MSigDB v2024.1.Hs (Liberzon et al. 2015, Cell Systems).
        https://www.gsea-msigdb.org/gsea/msigdb/human/genesets.jsp?collection=H

The Hallmark gene sets themselves are public domain (gene symbols are facts).
The MSigDB curation is from the Broad Institute under their published license.

Gene lists below are subsets of the canonical leading-edge genes — sufficient
for hypothesis-conditioning in a POC. For exhaustive sets, fetch the full
MSigDB GMT separately.

Biological rationale for LumA vs LumB discrimination:

  LumA pole (low proliferation, ER-dependent):
    - HALLMARK_ESTROGEN_RESPONSE_EARLY: immediate ER target genes
    - HALLMARK_ESTROGEN_RESPONSE_LATE:  delayed ER target genes

  LumB pole (high proliferation, cell-cycle-driven):
    - HALLMARK_E2F_TARGETS:    DNA replication / cell cycle
    - HALLMARK_G2M_CHECKPOINT: mitotic checkpoint / G2->M
    - HALLMARK_MYC_TARGETS_V1: MYC-regulated growth / metabolism
"""
from __future__ import annotations

from dataclasses import dataclass

# ----------------------------------------------------------------------------
# LumA-leaning sets (estrogen response — proliferation low, ER signaling high)
# ----------------------------------------------------------------------------

HALLMARK_ESTROGEN_RESPONSE_EARLY: tuple[str, ...] = (
    "ABCA3", "ABHD2", "ADCY1", "ADCY9", "AFF1", "AGR2", "AMFR", "ANXA9",
    "AR", "AREG", "BAG1", "BCL2", "BHLHE40", "CA12", "CALB2", "CALCR",
    "CCND1", "CD44", "CELSR1", "CELSR2", "CHST8", "CISH", "CLDN7", "CXCL12",
    "CYP26B1", "DEPTOR", "DHRS2", "DLC1", "EGR3", "ELF1", "ELOVL2", "ELOVL5",
    "ESR1", "FASN", "FCMR", "FHL2", "FKBP4", "FLNB", "FOS", "FOXA1",
    "FOXC1", "GATA3", "GFRA1", "GREB1", "HR", "HSPB8", "IGF1R", "IGFBP4",
    "INPP5F", "KCNK15", "KLF4", "KRT13", "KRT15", "KRT18", "KRT19", "KRT8",
    "MLPH", "MREG", "MSMB", "MUC1", "MYB", "MYBBP1A", "MYC", "NPY1R",
    "NRIP1", "OLFM1", "P2RY2", "PDLIM3", "PDZK1", "PGR", "PMAIP1", "PODXL",
    "PSAT1", "PTGES", "RAB17", "RAB31", "RASGRP1", "RET", "RHOBTB3", "SCNN1A",
    "SGK1", "SGK3", "SLC1A1", "SLC1A4", "SLC22A5", "SLC27A2", "SLC39A6", "SLC7A5",
    "STC2", "SVIL", "SYBU", "SYT12", "TFAP2C", "TFF1", "TFF3", "TGM2",
    "THSD4", "TIAM1", "TIPARP", "TMPRSS3", "TOB1", "TPBG", "TPD52L1", "TUBB2B",
    "UGCG", "UNC119", "WFDC2", "XBP1", "ZNF185",
)

HALLMARK_ESTROGEN_RESPONSE_LATE: tuple[str, ...] = (
    "ABAT", "ABCA3", "ABHD2", "ACOX2", "ADCY9", "AFF1", "AGR2", "AHCY",
    "AHNAK", "AKAP1", "ALCAM", "ALDH3A2", "AR", "AREG", "ASB13", "ASCL1",
    "ATP2B4", "BAG1", "BATF", "BCAS3", "BCL2", "BLVRB", "CACNA2D2", "CALB2",
    "CCND1", "CD44", "CDC14B", "CELSR2", "CELL", "CHST8", "CKB", "CLIC3",
    "CLU", "CYP1A1", "CYP26A1", "DCXR", "DEPTOR", "DHRS2", "DUSP4", "ELF3",
    "ELF5", "ELOVL5", "ESR1", "EZR", "FAM102A", "FAM134B", "FARP1", "FASN",
    "FCMR", "FGFR3", "FKBP4", "FLNB", "FOXC1", "GFRA1", "GLA", "GREB1",
    "ID2", "IGSF1", "IL17RB", "JAK2", "KCNK5", "KLF4", "KRT13", "KRT15",
    "KRT18", "KRT19", "KRT8", "MAPT", "MED13L", "MLPH", "MPPED2", "MUC1",
    "MYB", "MYC", "NBL1", "NPY1R", "NRIP1", "OPN3", "OVOL2", "PDLIM3",
    "PDZK1", "PGR", "PLAC1", "PMAIP1", "PRLR", "PRSS23", "PTGER3", "PTGES",
    "RAB31", "RBBP8", "RET", "S100A1", "SGK1", "SGK3", "SLC1A1", "SLC1A4",
    "SLC24A3", "SLC26A2", "SLC27A2", "SLC7A5", "STC2", "SUSD3", "SYT12",
    "TFF1", "TFF3", "THSD4", "TIAM1", "TIPARP", "TJP3", "TMPRSS3", "TOB1",
    "TPBG", "TPD52L1", "TSPAN1", "UGCG", "UGDH", "XBP1", "ZBTB16",
)

# ----------------------------------------------------------------------------
# LumB-leaning sets (proliferation — cell cycle, mitotic, MYC-driven growth)
# ----------------------------------------------------------------------------

HALLMARK_E2F_TARGETS: tuple[str, ...] = (
    "AK2", "ANP32E", "ASF1A", "ASF1B", "ATAD2", "AURKA", "AURKB", "BARD1",
    "BIRC5", "BRCA1", "BRCA2", "BRIP1", "BUB1B", "CBX5", "CCNB2", "CCNE1",
    "CCP110", "CDC20", "CDC25A", "CDC25B", "CDC45", "CDC6", "CDCA3", "CDCA8",
    "CDK1", "CDK4", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2C", "CDKN3", "CENPE",
    "CENPM", "CHEK1", "CHEK2", "CIT", "CKS1B", "CKS2", "CNOT9", "CSE1L",
    "CTCF", "CTPS1", "DCK", "DCLRE1B", "DCTPP1", "DDX39A", "DEK", "DEPDC1",
    "DIAPH3", "DLGAP5", "DNMT1", "DONSON", "DSCC1", "DUT", "E2F1", "E2F2",
    "E2F8", "EED", "EIF2S1", "ESPL1", "EXOSC8", "EZH2", "FOXM1", "GINS1",
    "GINS3", "GINS4", "GSPT1", "H2AX", "HELLS", "HMGA1", "HMGB2", "HMGB3",
    "HMMR", "HUS1", "ILF3", "ING3", "IPO7", "JPT1", "KIF18B", "KIF22",
    "KIF2C", "KIF4A", "KPNA2", "LBR", "LIG1", "LMNB1", "LUC7L3", "LYAR",
    "MAD2L1", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "MELK",
    "MKI67", "MLH1", "MMS22L", "MRE11", "MSH2", "MTHFD2", "MXD3", "MYBL2",
    "MYC", "NAA38", "NASP", "NBN", "NCAPD2", "NME1", "NOLC1", "NOP56",
    "NUP107", "NUP153", "NUP205", "ORC2", "ORC6", "PA2G4", "PAICS", "PAN2",
    "PCNA", "PDS5B", "PHF5A", "PLK1", "PLK4", "PMS2", "PNN", "POLA2",
    "POLD1", "POLD2", "POLD3", "POLE", "POLE4", "POP7", "PPM1D", "PPP1R8",
    "PRDX4", "PRIM2", "PRKDC", "PRPS1", "PSIP1", "PSMC3IP", "PTTG1", "RACGAP1",
    "RAD1", "RAD21", "RAD50", "RAD51AP1", "RAD51C", "RAN", "RANBP1", "RBBP7",
    "RFC1", "RFC2", "RFC3", "RNASEH2A", "RPA1", "RPA2", "RPA3", "RRM2",
    "SHMT1", "SLBP", "SMC1A", "SMC3", "SMC4", "SMC6", "SNRPB", "SPAG5",
    "SPC24", "SPC25", "SRSF1", "SRSF2", "SSRP1", "STAG1", "STMN1", "SUV39H1",
    "SYNCRIP", "TACC3", "TBRG4", "TCF19", "TFRC", "TIMELESS", "TIPIN", "TK1",
    "TMPO", "TOP2A", "TP53", "TRA2B", "TRIP13", "TUBB", "TUBG1", "UBE2S",
    "UBE2T", "UBR7", "UNG", "USP1", "WDR90", "WEE1", "XPO1", "XRCC6",
    "ZW10",
)

HALLMARK_G2M_CHECKPOINT: tuple[str, ...] = (
    "ARID4A", "ATF5", "ATRX", "AURKA", "AURKB", "BARD1", "BCL3", "BIRC5",
    "BRCA2", "BUB1", "BUB3", "CASP8AP2", "CBX1", "CCNA2", "CCNB2", "CCND1",
    "CCNE1", "CCNF", "CCNT1", "CDC20", "CDC25A", "CDC25B", "CDC27", "CDC45",
    "CDC6", "CDC7", "CDK1", "CDK4", "CDKN1B", "CDKN2C", "CDKN3", "CENPA",
    "CENPE", "CENPF", "CFLAR", "CHAF1A", "CHEK1", "CHMP1A", "CKAP2", "CKAP5",
    "CKS1B", "CKS2", "CTCF", "CUL1", "CUL3", "CUL4A", "CUL5", "DBF4",
    "DDX39A", "DKC1", "DMD", "DTYMK", "E2F1", "E2F2", "E2F3", "E2F4",
    "EFNA5", "EGF", "ESPL1", "EWSR1", "EZH2", "FANCC", "FBRSL1", "FOXN3",
    "G3BP1", "GINS2", "GSPT1", "H2AX", "H2AZ1", "HIF1A", "HIRA", "HMGA1",
    "HMGB3", "HMGN2", "HMMR", "HNRNPD", "HNRNPU", "HOXC10", "HSPA8", "HUS1",
    "ILF3", "INCENP", "JPT1", "KATNA1", "KIF11", "KIF15", "KIF20B", "KIF22",
    "KIF23", "KIF2C", "KIF4A", "KIF5B", "KIFC1", "KMT5A", "KNL1", "KPNA2",
    "KPNB1", "LBR", "LIG3", "LMNB1", "MAD2L1", "MAPK14", "MARCKS", "MCM2",
    "MCM3", "MCM5", "MCM6", "MEIS1", "MEIS2", "MKI67", "MNAT1", "MT2A",
    "MTF2", "MYBL2", "MYC", "NASP", "NCL", "NDC80", "NEK2", "NOLC1",
    "NOTCH2", "NSD2", "NUMA1", "NUP50", "NUP98", "NUSAP1", "ODF2", "ODC1",
    "ORC5", "ORC6", "PAFAH1B1", "PBK", "PDS5B", "PLK1", "PLK4", "POLA2",
    "POLE", "POLQ", "PRC1", "PRIM2", "PRKDC", "PRMT5", "PRPF4B", "PTTG1",
    "PTTG3P", "PURA", "RACGAP1", "RAD21", "RAD23B", "RASAL2", "RBBP7", "RBL1",
    "RNF8", "RPA2", "RPA3", "RPS6KA5", "SAP30", "SFPQ", "SLC12A2", "SLC38A1",
    "SLC7A1", "SLC7A5", "SMAD3", "SMARCC1", "SMC1A", "SMC2", "SMC4", "SNRPD1",
    "SQLE", "SRSF1", "SRSF10", "SRSF2", "STAG1", "STIL", "STMN1", "SUV39H1",
    "SYNCRIP", "TACC3", "TENT4A", "TFDP1", "TGFB1", "TLE3", "TMPO", "TNPO2",
    "TOP1", "TOP2A", "TPX2", "TRA2B", "TRAIP", "TROAP", "TTK", "UBE2C",
    "UBE2S", "UCK2", "UPF1", "WRN", "XPO1", "YTHDC1",
)

HALLMARK_MYC_TARGETS_V1: tuple[str, ...] = (
    "ABCE1", "ACP1", "AIMP2", "AP3S1", "APEX1", "BUB3", "C1QBP", "CAD",
    "CANX", "CBX3", "CCNA2", "CCT2", "CCT3", "CCT4", "CCT5", "CCT7",
    "CDC20", "CDC45", "CDK2", "CDK4", "CLNS1A", "CNBP", "COPS5", "COX5A",
    "CSTF2", "CTPS1", "CUL1", "CYC1", "DDX18", "DDX21", "DEK", "DHX15",
    "DUT", "EEF1B2", "EIF2S1", "EIF2S2", "EIF3B", "EIF3D", "EIF3J", "EIF4A1",
    "EIF4E", "EIF4G2", "EIF4H", "EPRS1", "ERH", "ETF1", "EXOSC7", "FAM120A",
    "FBL", "G3BP1", "GLO1", "GNL3", "GOT2", "GSPT1", "H2AZ1", "HDAC2",
    "HDDC2", "HDGF", "HNRNPA1", "HNRNPA2B1", "HNRNPA3", "HNRNPC", "HNRNPD", "HNRNPR",
    "HNRNPU", "HPRT1", "HSP90AB1", "HSPD1", "HSPE1", "IARS1", "IFRD1", "ILF2",
    "IMPDH2", "KARS1", "KPNA2", "KPNB1", "LDHA", "LSM2", "LSM7", "M6PR",
    "MAD2L1", "MCM2", "MCM4", "MCM5", "MCM6", "MCM7", "MRPL23", "MRPL9",
    "MRPS18B", "MYC", "NAP1L1", "NCBP1", "NCBP2", "NDUFAB1", "NHP2", "NME1",
    "NOLC1", "NOP16", "NOP56", "NPM1", "ODC1", "ORC2", "PA2G4", "PABPC1",
    "PABPC4", "PCBP1", "PCNA", "PGK1", "PHB1", "PHB2", "POLD2", "POLE3",
    "POP7", "PPIA", "PPM1G", "PRDX3", "PRDX4", "PRPF31", "PRPS2", "PSMA1",
    "PSMA2", "PSMA4", "PSMA6", "PSMA7", "PSMB2", "PSMB3", "PSMC4", "PSMC6",
    "PSMD1", "PSMD14", "PSMD3", "PSMD7", "PSMD8", "PTGES3", "PWP1", "RACK1",
    "RAD23B", "RAN", "RANBP1", "RFC4", "RNPS1", "RPL14", "RPL18", "RPL22",
    "RPL34", "RPL6", "RPLP0", "RPS10", "RPS2", "RPS3", "RPS5", "RPS6",
    "RRM1", "RRP9", "RSL1D1", "RUVBL2", "SERBP1", "SET", "SF3A1", "SF3B3",
    "SLC25A3", "SMARCC1", "SNRPA", "SNRPA1", "SNRPB2", "SNRPD1", "SNRPD2", "SNRPD3",
    "SNRPG", "SRM", "SRPK1", "SRSF1", "SRSF2", "SRSF3", "SRSF7", "SSB",
    "SSBP1", "STARD7", "SYNCRIP", "TARDBP", "TCP1", "TFDP1", "TOMM70", "TRA2B",
    "TRIM28", "TUFM", "TXNL4A", "TYMS", "U2AF1", "UBA2", "UBE2E1", "UBE2L3",
    "USP1", "VBP1", "VDAC1", "VDAC3", "XPO1", "XPOT", "XRCC6", "YWHAE",
    "YWHAQ",
)

# ----------------------------------------------------------------------------
# Bundled registry
# ----------------------------------------------------------------------------

POLE_LUMA: tuple[str, ...] = (
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
)
POLE_LUMB: tuple[str, ...] = (
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
)

# v0.9: pole-pair for Luminal-lineage vs Basal-lineage classification.
# These pole names reference Hallmark sets that are NOT in priors.HALLMARK_SETS,
# so callers must pass the full 50-set catalog from
# `dmoi_brca.hallmark.load_hallmark_gmt(...)` to `make_pole_masks(...)` via the
# `hallmark_sets=` kwarg. Without that override these names will raise KeyError
# (intentional: forces the caller to be explicit about which catalog is in use).
POLE_LUMINAL: tuple[str, ...] = (
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
    "HALLMARK_ANDROGEN_RESPONSE",
)
POLE_BASAL: tuple[str, ...] = (
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_G2M_CHECKPOINT",
)

HALLMARK_SETS: dict[str, tuple[str, ...]] = {
    "HALLMARK_ESTROGEN_RESPONSE_EARLY": HALLMARK_ESTROGEN_RESPONSE_EARLY,
    "HALLMARK_ESTROGEN_RESPONSE_LATE": HALLMARK_ESTROGEN_RESPONSE_LATE,
    "HALLMARK_E2F_TARGETS": HALLMARK_E2F_TARGETS,
    "HALLMARK_G2M_CHECKPOINT": HALLMARK_G2M_CHECKPOINT,
    "HALLMARK_MYC_TARGETS_V1": HALLMARK_MYC_TARGETS_V1,
}


@dataclass(frozen=True)
class GeneSetProjection:
    """Result of projecting a gene set onto a feature matrix's gene column space."""
    name: str
    genes_in_set: int                 # Total genes in the hallmark set
    genes_in_features: int            # How many overlap the feature space
    feature_indices: tuple[int, ...]  # Column indices in feature matrix
    matched_genes: tuple[str, ...]    # Symbols of matched genes (subset of set)
    missing_genes: tuple[str, ...]    # Symbols in set but not in features

    @property
    def overlap_fraction(self) -> float:
        return self.genes_in_features / self.genes_in_set if self.genes_in_set else 0.0


def project_to_features(
    set_name: str,
    feature_symbols: list[str] | tuple[str, ...],
) -> GeneSetProjection:
    """Project a Hallmark gene set onto a feature matrix's gene symbol space.

    Args:
        set_name:         Hallmark set key (must be in HALLMARK_SETS).
        feature_symbols:  Gene symbols corresponding to RNA-seq feature columns.

    Returns:
        GeneSetProjection with feature_indices ready to index a feature matrix.
    """
    if set_name not in HALLMARK_SETS:
        raise KeyError(
            f"Unknown gene set: {set_name}. "
            f"Available: {sorted(HALLMARK_SETS)}",
        )
    set_genes = HALLMARK_SETS[set_name]
    feature_lookup = {sym: i for i, sym in enumerate(feature_symbols)}
    matched: list[str] = []
    indices: list[int] = []
    missing: list[str] = []
    for gene in set_genes:
        if gene in feature_lookup:
            matched.append(gene)
            indices.append(feature_lookup[gene])
        else:
            missing.append(gene)
    return GeneSetProjection(
        name=set_name,
        genes_in_set=len(set_genes),
        genes_in_features=len(matched),
        feature_indices=tuple(indices),
        matched_genes=tuple(matched),
        missing_genes=tuple(missing),
    )


def project_pole(
    pole_sets: tuple[str, ...],
    feature_symbols: list[str] | tuple[str, ...],
) -> dict[str, GeneSetProjection]:
    """Project all Hallmark sets defining a pole (LumA or LumB) onto features."""
    return {name: project_to_features(name, feature_symbols) for name in pole_sets}

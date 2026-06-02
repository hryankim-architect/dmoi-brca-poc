# MSigDB Hallmark gene sets

## File

`h.all.v2024.1.Hs.symbols.gmt`, the 50 MSigDB Hallmark gene sets,
human gene symbols, MSigDB release v2024.1.Hs.

- 50 sets
- Format: GMT (tab-separated, one set per line:
  `set_name<TAB>description_url<TAB>gene1<TAB>gene2<TAB>...`)
- Source: <https://www.gsea-msigdb.org/gsea/msigdb/>
- Direct URL used at fetch time:
  `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt`
- Fetched: 2026-05-28
- File size: 48,690 bytes
- File `last-modified` (Broad CDN): 2024-08-09

## License

The contents of MSigDB v6.0+ and v2022.1+ are
**Copyright (c) 2004-2025 Broad Institute, Inc., Massachusetts
Institute of Technology, and Regents of the University of California**,
distributed under the
[Creative Commons Attribution 4.0 International License (CC-BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

The Hallmark collection is not subject to additional KEGG / BioCarta
restrictions (see the MSigDB license terms page for the prefix list of
restricted subcollections).

## Reproduce

```bash
curl -sSL \
  https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt \
  -o data/msigdb/h.all.v2024.1.Hs.symbols.gmt
```

## How this repo uses it

`src/dmoi_brca/hallmark.py` parses the gmt file into a
`dict[str, list[str]]` (set_name → gene list). The 50-set rollup is
the v0.6 expansion of the v0.5 pathway-IG aggregation: v0.5 loaded
only the 5 Hallmark sets that were already in `priors.py` for the pole
masks; v0.6 loads all 50 to verify the v0.5 top-pathway finding wasn't
an artifact of the restricted set list.

## Citation

Liberzon A, Birger C, Thorvaldsdóttir H, Ghandi M, Mesirov JP,
Tamayo P. *The Molecular Signatures Database (MSigDB) hallmark gene
set collection.* Cell Systems. 2015;1(6):417-425.
[doi:10.1016/j.cels.2015.12.004](https://doi.org/10.1016/j.cels.2015.12.004)

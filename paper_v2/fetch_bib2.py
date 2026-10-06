#!/usr/bin/env python3
"""CrossRef-based verified BibTeX fetch (DBLP unreachable from this host).
For each key: search works?query.bibliographic, require title token + year match,
then fetch publisher-verified BibTeX via doi.org content negotiation."""
import json, re, subprocess, os
from pathlib import Path

env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
env["PATH"] = os.environ["PATH"]
OUT = Path(__file__).parent / "fetched2.bib"

# key: (query, required title substrings (lower), expected year)
QUERIES = {
    "gresova2023genomic": ("Genomic Benchmarks deep learning", ["genomic benchmark"], 2023),
    "notin2023proteingym": ("ProteinGym massively scale protein mutation effects", ["proteingym"], 2023),
    "dealmeida2022deepstarr": ("DeepSTARR predicting sequence activity enhancer", ["deepstarr"], 2022),
    "ray2013rnacompete": ("compendium RNA-binding motifs decoding gene regulation", ["rna-binding motifs"], 2013),
    "dallago2024flip": ("FLIP benchmark protein fitness landscape machine learning", ["flip"], 2024),
    "zhou2015deepsea": ("predicting effects of noncoding variants deepsea", ["deepsea"], 2015),
    "xu2022peer": ("PEER computational pre-training exon recognition", ["peer"], 2022),
    "lin2023esm2": ("evolutionary-scale prediction atomic-level protein structure language model", ["evolutionary-scale"], 2023),
    "leng2024vcd": ("mitigating object hallucinations large vision-language models visual contrastive decoding", ["contrastive decoding"], 2024),
    "li2023contrastive": ("contrastive decoding open-ended text generation optimization", ["contrastive decoding"], 2023),
    "bouthillier2021variance": ("accounting for variance machine learning benchmarks", ["variance"], 2021),
}

def curl(args):
    return subprocess.run(["curl", "-s", "--max-time", "30"] + args,
                          capture_output=True, env=env).stdout.decode("utf-8", "replace")

entries, missing = [], []
for key, (q, toks, year) in QUERIES.items():
    j = json.loads(curl(["https://api.crossref.org/works?query.bibliographic="
                         + q.replace(" ", "+") + "&rows=8"]) or "{}")
    items = (j.get("message") or {}).get("items", [])
    chosen = None
    for it in items:
        t = (it.get("title") or [""])[0].lower()
        y = (it.get("published-print") or it.get("published-online") or
             it.get("issued") or {}).get("date-parts", [[None]])[0][0]
        if all(k in t for k in toks) and y == year:
            chosen = it
            break
    if not chosen:
        missing.append((key, "no-title/year-match"))
        continue
    doi = chosen["DOI"]
    bib = curl(["-LH", "Accept: application/x-bibtex", f"https://doi.org/{doi}"]).strip()
    m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", bib)
    if "@" not in bib or not m:
        missing.append((key, "bibtex-fetch-fail"))
        continue
    bib = bib[:m.start()] + f"@{m.group(1)}{{{key}," + bib[m.end():]
    entries.append(f"% verified via CrossRef DOI {doi} for {key}\n" + bib.strip() + "\n")
    print(f"OK  {key}  <- {doi}")

OUT.write_text("\n".join(entries))
print("MISSING:", missing)

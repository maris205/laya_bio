#!/usr/bin/env python3
"""Pass 2: fix wrong-venue entries and fetch the remaining five."""
import json, re, subprocess, os
from pathlib import Path

env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
env["PATH"] = os.environ["PATH"]

def curl(args):
    return subprocess.run(["curl", "-s", "--max-time", "30"] + args,
                          capture_output=True, env=env).stdout.decode("utf-8", "replace")

def rename(bib, key):
    m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", bib)
    return bib[:m.start()] + f"@{m.group(1)}{{{key}," + bib[m.end():]

def doi_bib(key, doi):
    bib = curl(["-LH", "Accept: application/x-bibtex", f"https://doi.org/{doi}"]).strip()
    if "@" not in bib:
        print(f"FAIL {key} {doi}")
        return None
    print(f"OK   {key} <- {doi}")
    return rename(bib, key)

def arxiv_bib(key, aid):
    xml = curl([f"https://export.arxiv.org/api/query?id_list={aid}"])
    t = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    if not t:
        print(f"FAIL {key} arxiv {aid}")
        return None
    title = re.sub(r"\s+", " ", t.group(1)).strip()
    auth = re.findall(r"<name>(.*?)</name>", xml)
    year = re.search(r"<published>(\d{4})", xml)
    return (f"@misc{{{key},\n  title = {{{title}}},\n  author = {{{' and '.join(auth)}}},\n"
            f"  year = {{{year.group(1) if year else ''}}},\n  eprint = {{{aid}}},\n"
            "  archivePrefix = {arXiv},\n  note = {preprint}}}\n")

out = []
# corrected: RNAcompete original Nat Biotech 2013
out.append(doi_bib("ray2013rnacompete", "10.1038/nbt.2515"))
# ProteinGym: published NeurIPS 2023 -> use arXiv record (no journal DOI)
out.append(arxiv_bib("notin2023proteingym", "2310.08900"))
# remaining five via CrossRef search
SEARCH = {
    "gresova2023genomic": ("Genomic Benchmarks for Deep Learning", ["genomic benchmarks"], 2023),
    "dallago2024flip": ("FLIP benchmark protein fitness landscape prediction", ["fitness landscape"], 2024),
    "zhou2015deepsea": ("Predicting effects of noncoding variants deep learning sequence model", ["noncoding variants"], 2015),
    "xu2022peer": ("PEER computational pre-training exon recognition network across species", ["exon recognition"], 2022),
    "bouthillier2021variance": ("Accounting for variance in machine learning benchmarks", ["variance"], 2021),
}
for key, (q, toks, year) in SEARCH.items():
    j = json.loads(curl(["https://api.crossref.org/works?query.bibliographic="
                         + q.replace(" ", "+") + "&rows=10"]) or "{}")
    hit = None
    for it in (j.get("message") or {}).get("items", []):
        t = (it.get("title") or [""])[0].lower()
        y = (it.get("published-print") or it.get("published-online") or
             it.get("issued") or {}).get("date-parts", [[None]])[0][0]
        if all(k in t for k in toks) and y == year:
            hit = it
            break
    if hit:
        out.append(doi_bib(key, hit["DOI"]))
    else:
        print(f"MISS {key}")

Path("fetched3.bib").write_text("\n".join(x for x in out if x))

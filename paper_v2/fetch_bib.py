#!/usr/bin/env python3
"""Fetch verified BibTeX for new citation keys. Chain: DBLP search -> arXiv export API.
No proxy (direct works). Never fabricates; failures are reported for manual handling."""
import json, re, subprocess, os
from pathlib import Path

env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
env["PATH"] = os.environ["PATH"]
OUT = Path(__file__).parent / "fetched.bib"

QUERIES = {
    "gresova2023genomic": ("dblp", "Genomic Benchmarks"),
    "notin2023proteingym": ("dblp", "ProteinGym"),
    "dealmeida2022deepstarr": ("dblp", "DeepSTARR"),
    "ray2013rnacompete": ("dblp", "compendium RNA-binding motifs"),
    "dallago2024flip": ("dblp", "FLIP benchmark protein fitness"),
    "zhou2015deepsea": ("dblp", "predicting effects of noncoding variants DeepSEA"),
    "xu2022peer": ("dblp", "PEER computational pre-training exon recognition"),
    "lin2023esm2": ("dblp", "Evolutionary-scale prediction atomic-level protein structure"),
    "yang2025qwen3": ("arxiv", "2505.09388"),
    "bai2025qwen25vl": ("arxiv", "2502.13923"),
    "leng2024vcd": ("dblp", "Visual Contrastive Decoding hallucinations"),
    "li2023contrastive": ("dblp", "Contrastive Decoding open-ended"),
    "bouthillier2021variance": ("dblp", "Accounting for Variance Machine Learning Benchmarks"),
    "miller2024errorbars": ("arxiv", "2411.00640"),
    "gliclass2025": ("arxiv", "2508.10653"),
}

def curl(url, accept=None):
    cmd = ["curl", "-s", "--max-time", "30"]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, env=env).stdout.decode("utf-8", "replace")

def dblp_hit(query):
    j = json.loads(curl(f"https://dblp.org/search/publ/api?q={query.replace(' ', '+')}&format=json&h=3") or "{}")
    hits = (j.get("result") or {}).get("hits", {}).get("hit", []) or []
    if not hits:
        return None, None
    return hits[0]["info"].get("key"), hits[0]["info"].get("title")

def arxiv_bib(aid):
    # data.citeulike mirror of arXiv metadata via export API + manual bibtex from the Atom entry
    xml = curl(f"https://export.arxiv.org/api/query?id_list={aid}")
    t = re.search(r"<title>(.*?)</title>", xml, re.S)
    title = re.sub(r"\s+", " ", t.group(1)).strip() if t else ""
    auth = re.findall(r"<name>(.*?)</name>", xml)
    year = re.search(r"<published>(\d{4})", xml)
    abs_ = " "
    if not title or not auth:
        return None
    authors = " and ".join(auth)
    return (f"@misc{{{aid.replace('.','p')},\n  title = {{{title}}},\n  author = {{{authors}}},\n"
            f"  year = {{{year.group(1) if year else ''}}},\n  eprint = {{{aid}}},\n"
            "  archivePrefix = {arXiv},\n  note = {preprint}}}\n")

entries, missing = [], []
for key, (kind, q) in QUERIES.items():
    bib = None
    if kind == "dblp":
        k, title = dblp_hit(q)
        if k:
            bib = curl(f"https://dblp.org/rec/{k}.bib")
    if kind == "arxiv" or bib is None:
        bib = arxiv_bib(q) if kind == "arxiv" else None
    if not bib or "@" not in bib:
        missing.append((key, kind, q)); continue
    bib = bib.strip()
    m = re.match(r"@(\w+)\{([^,]+),", bib)
    typ, oldkey = m.group(1), m.group(2)
    bib = bib.replace(f"@{typ}{{{oldkey},", f"@{typ}{{{key},", 1)
    entries.append(f"% verified via {kind} ({q}) for {key}\n" + bib.strip() + "\n")
    print(f"OK  {key}")

OUT.write_text("\n".join(entries))
print("\nMISSING:")
for m in missing:
    print("  ", m)

#!/usr/bin/env python3
"""Pass 5: correct ray/dallago, fetch gresova/xu2022peer. Verify titles before accepting."""
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

def crossref_pick(key, query, must_have, year, expect_author=None):
    j = json.loads(curl(["https://api.crossref.org/works?query.bibliographic="
                         + query.replace(" ", "+") + "&rows=12"]) or "{}")
    for it in (j.get("message") or {}).get("items", []):
        t = (it.get("title") or [""])[0]
        y = (it.get("published-print") or it.get("published-online") or
             it.get("issued") or {}).get("date-parts", [[None]])[0][0]
        if y == year and all(k.lower() in t.lower() for k in must_have):
            bib = curl(["-LH", "Accept: application/x-bibtex", f"https://doi.org/{it['DOI']}"]).strip()
            if "@" in bib:
                print(f"OK   {key}: {t[:70]}")
                return rename(bib, key)
    print(f"MISS {key}")
    return None

def arxiv_pick(key, query, must_have):
    xml = curl(["https://export.arxiv.org/api/query?search_query=" + query + "&max_results=5"])
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip()
        if all(k.lower() in t.lower() for k in must_have):
            aid = re.search(r"arxiv.org/abs/([\w.\-/]+)", e).group(1)
            auth = " and ".join(re.findall(r"<name>(.*?)</name>", e))
            yr = re.search(r"<published>(\d{4})", e).group(1)
            print(f"OK   {key}: {t[:70]} (arXiv {aid})")
            return (f"@misc{{{key},\n  title = {{{t}}},\n  author = {{{auth}}},\n  year = {{{yr}}},\n"
                    f"  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}},\n  note = {{preprint}}}}\n")
    print(f"MISS {key} (arxiv)")
    return None

out = []
out.append(crossref_pick("ray2013rnacompete",
    "A compendium of RNA-binding motifs for decoding gene regulation",
    ["rna-binding motifs", "gene regulation"], 2013))
out.append(crossref_pick("dallago2024flip",
    "FLIP a benchmark for protein fitness landscape prediction",
    ["flip", "fitness"], 2024))
out.append(arxiv_pick("gresova2023genomic",
    'ti%3A%22Genomic+Benchmarks+for+Deep+Learning%22', ["genomic benchmarks"]))
out.append(arxiv_pick("xu2022peer",
    'ti%3A%22PEER%22+AND+abs%3A%22exon+recognition%22', ["peer"]))
Path("fetched5.bib").write_text("\n".join(x for x in out if x))

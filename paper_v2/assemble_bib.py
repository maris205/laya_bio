#!/usr/bin/env python3
"""Assemble references.bib: old verified entries + fetched corrections.
Every new entry is verified against CrossRef/arXiv metadata; titles are checked."""
import json, re, subprocess, os
from pathlib import Path

env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
env["PATH"] = os.environ["PATH"]
HERE = Path(__file__).parent

PROXY_ENV = dict(os.environ, http_proxy="http://127.0.0.1:7890", https_proxy="http://127.0.0.1:7890")

def curl(args):
    r = subprocess.run(["curl", "-s", "--max-time", "25"] + args,
                       capture_output=True, env=env).stdout.decode("utf-8", "replace")
    if not r.strip():  # direct failed -> clash proxy retry
        r = subprocess.run(["curl", "-s", "--max-time", "25"] + args,
                           capture_output=True, env=PROXY_ENV).stdout.decode("utf-8", "replace")
    return r

def arxiv_entry(key, aid, must_have):
    xml = curl([f"https://export.arxiv.org/api/query?id_list={aid}"])
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip()
        if all(k.lower() in t.lower() for k in must_have):
            auth = " and ".join(re.findall(r"<name>(.*?)</name>", e))
            yr = re.search(r"<published>(\d{4})", e).group(1)
            print(f"OK  {key}: {t[:70]}")
            return (f"@misc{{{key},\n  title = {{{t}}},\n  author = {{{auth}}},\n  year = {{{yr}}},\n"
                    f"  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}}}}\n")
    print(f"FAIL {key}: expected {must_have}, got feed")
    return None

def arxiv_search_entry(key, sq, must_have):
    xml = curl([f"https://export.arxiv.org/api/query?search_query={sq}&max_results=5"])
    return next((arxiv_entry(key, re.search(r"arxiv.org/abs/([\w.\-/]+)", e).group(1), must_have)
                 for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S)
                 if all(k.lower() in re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).lower() for k in must_have)),
                None)

def crossref_entry(key, query, must_have, year, first_author=None):
    j = json.loads(curl(["https://api.crossref.org/works?query.bibliographic="
                         + query.replace(" ", "+") + "&rows=12"]) or "{}")
    for it in (j.get("message") or {}).get("items", []):
        t = (it.get("title") or [""])[0]
        y = (it.get("published-print") or it.get("published-online") or
             it.get("issued") or {}).get("date-parts", [[None]])[0][0]
        if y != year or not all(k.lower() in t.lower() for k in must_have):
            continue
        if t.lower().startswith("faculty opinions"):
            continue
        if first_author:
            au = ((it.get("author") or [{}])[0].get("family") or "").lower()
            if first_author.lower() not in au:
                print(f"skip {key}: first author {au} != {first_author}")
                continue
        bib = curl(["-LH", "Accept: application/x-bibtex", f"https://doi.org/{it['DOI']}"]).strip()
        m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", bib)
        if not m:
            continue
        title = re.search(r"title\s*=\s*\{(.*)\}", bib)
        print(f"OK  {key}: {title.group(1)[:70] if title else '?'}")
        return bib[:m.start()] + f"@{m.group(1)}{{{key}," + bib[m.end():]
    print(f"FAIL {key}")
    return None

new = {}
# --- arXiv (title-checked) ---
new["yang2025qwen3"] = arxiv_entry("yang2025qwen3", "2505.09388", ["qwen3"])
new["bai2025qwen25vl"] = arxiv_entry("bai2025qwen25vl", "2502.13923", ["qwen2.5-vl"])
new["miller2024errorbars"] = arxiv_entry("miller2024errorbars", "2411.00640", ["error bar"])
new["gliclass2025"] = arxiv_search_entry("gliclass2025", 'ti%3A%22GLiClass%22', ["gliclass"])
new["bouthillier2021variance"] = arxiv_search_entry(
    "bouthillier2021variance", 'ti%3A%22Accounting+for+Variance+in+Machine+Learning+Benchmarks%22',
    ["variance"])
# --- CrossRef (title+year+first-author checked) ---
new["ray2013rnacompete"] = crossref_entry("ray2013rnacompete",
    "A compendium of RNA-binding motifs for decoding gene regulation",
    ["rna-binding motifs", "gene regulation"], 2013, first_author="Ray")
new["gresova2022genomic"] = crossref_entry("gresova2022genomic",
    "Genomic Benchmarks Collection of Datasets for Genomic Sequence Classification",
    ["genomic benchmarks"], 2022, first_author="Gre")
new["dallago2021flip"] = crossref_entry("dallago2021flip",
    "FLIP Benchmark tasks in fitness landscape inference for proteins",
    ["flip", "fitness landscape"], 2021, first_author="Dallago")
new["notin2023proteingym"] = crossref_entry("notin2023proteingym",
    "ProteinGym Large-Scale Benchmarks Protein Design Fitness Prediction",
    ["proteingym"], 2023, first_author="Notin")
new["dealmeida2022deepstarr"] = crossref_entry("dealmeida2022deepstarr",
    "DeepSTARR predicts enhancer activity", ["deepstarr"], 2022, first_author="de Almeida")
new["zhou2015deepsea"] = crossref_entry("zhou2015deepsea",
    "Predicting effects of noncoding variants with deep learning-based sequence model",
    ["noncoding variants"], 2015, first_author="Zhou")
new["lin2023esm2"] = crossref_entry("lin2023esm2",
    "Evolutionary-scale prediction of atomic-level protein structure",
    ["evolutionary-scale"], 2023, first_author="Lin")
new["leng2024vcd"] = crossref_entry("leng2024vcd",
    "Mitigating Object Hallucinations Visual Contrastive Decoding",
    ["contrastive decoding"], 2024, first_author="Leng")
new["li2023contrastive"] = crossref_entry("li2023contrastive",
    "Contrastive Decoding: Open-ended Text Generation as Optimization",
    ["contrastive decoding"], 2023, first_author="Li")
# community repos (verified via GitHub API earlier this session)
new["nanojev2026"] = None   # kept from old_verified.bib
new["semif2025"] = None

missing = [k for k, v in new.items() if v is None]
(HERE / "new_fetched.bib").write_text("\n".join(v for v in new.values() if v))
print("\nFAILED:", missing)

# assemble final references.bib
old = (HERE / "old_verified.bib").read_text()
cited = set()
for f in (HERE / "sections").glob("*.tex"):
    for m in re.finditer(r"\\cite[tp]*\**(?:\[[^\]]*\])*\{([^}]*)\}", f.read_text()):
        cited.update(k.strip() for k in m.group(1).split(",") if k.strip())
# key renames from the draft pass
renames = {"gresova2023genomic": "gresova2022genomic", "dallago2024flip": "dallago2021flip"}
cited = {renames.get(k, k) for k in cited}
have = set(re.findall(r"(?m)^@\w+\{([^,]+),", old)) | set(re.findall(r"(?m)^@\w+\{([^,]+),", (HERE / "new_fetched.bib").read_text()))
print("cited but missing:", sorted(cited - have))
(HERE / "references.bib").write_text(
    "% BioDecisionBench references — every entry verified (old bib or CrossRef/arXiv, 2026-10-06)\n\n"
    + old + "\n\n" + (HERE / "new_fetched.bib").read_text())
print("wrote references.bib with", len(have), "entries")

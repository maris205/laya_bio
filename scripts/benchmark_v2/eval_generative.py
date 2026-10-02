#!/usr/bin/env python3
"""#3 baseline: frozen causal-LM full-candidate-likelihood on benchmark v2.

For each noul/choice task, a small causal LM (Qwen3-0.6B, frozen) scores every
candidate by the length-normalized log-probability of the candidate text given
the prompt (question + sequence), and picks the argmax. Score tasks use the
per-task 5 anchor-value strings as candidates (approximating the level).

This is a FROZEN generative baseline — it is NOT fine-tuned on the benchmark, so
it is not "matched compute" with the trained #1/#2; it answers "can an
off-the-shelf small LM, via generative candidate likelihood, compete with the
trained typed-decision model on these biology tasks." Sequences are capped to a
char budget to bound prompt length (Qwen tokenizes raw DNA/protein inefficiently);
this truncation is a #3 limitation and is reported.
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

JEV = Path("/root/autodl-tmp/jev_gene")
OUT = JEV / "data/06_benchmark_v2_unified"
LM = JEV / "models/Qwen3-0.6B"
SEQ_CHAR_CAP = 120

def prompt(question, seq, modality):
    lab = "DNA" if modality == "dna" else ("RNA" if modality == "rna" else "Protein")
    return f"Question: {question}\n{lab} sequence: {seq[:SEQ_CHAR_CAP]}\nAnswer:"

@torch.no_grad()
def cand_logprob(model, tok, prompt_text, cand_text, dev):
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full = p_ids + tok(" " + cand_text, add_special_tokens=False)["input_ids"]
    n_c = len(full) - len(p_ids)
    if n_c <= 0:
        return float("-inf")
    inp = torch.tensor([full], device=dev)
    out = model(inp)
    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)  # predicting next token
    tgt = torch.tensor(full[1:], device=dev)
    tok_lp = logp.gather(1, tgt[:, None]).squeeze(-1)
    return float(tok_lp[-n_c:].sum().item()) / n_c  # length-normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--task-config", type=Path, default=JEV/"artifacts/benchmark_v2_train/bv2_full2/run_config.json")
    ap.add_argument("--max-per-task", type=int, default=100)
    ap.add_argument("--out", type=Path, default=JEV/"artifacts/benchmark_v2_eval/gen3")
    a = ap.parse_args()
    tasks = a.tasks or json.load(open(a.task_config))["tasks"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LM))
    model = AutoModelForCausalLM.from_pretrained(str(LM), torch_dtype=torch.bfloat16).to(dev).eval()
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for tk in tasks:
        d = OUT / tk
        f = d/"test.jsonl"
        if not f.exists(): f = d/"dev.jsonl"
        if not f.exists(): continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs: continue
        prim = recs[0]["primitive"]
        if prim == "multi_noul":
            results[tk] = {"skip": "multi_noul"}; continue
        if recs[0].get("sequences") and len(recs[0]["sequences"]) != 1:
            results[tk] = {"skip": "double_sequence"}; continue
        rng = np.random.default_rng(20261001)
        if len(recs) > a.max_per_task:
            recs = [recs[i] for i in rng.choice(len(recs), size=a.max_per_task, replace=False)]
        # candidate set + scoring
        if prim == "score":
            anchors = recs[0].get("anchors")
            if not anchors:
                results[tk] = {"skip": "no_anchors"}; continue
            cands = [f"{a:.6g}" for a in anchors]; gold_key = "gold_level"
        elif prim == "noul":
            cands = ["yes", "no"]; gold_key = "answer"  # candidates normalized to yes/no text
        else:
            cands = list(recs[0]["candidates"]); gold_key = "answer"
        ys, pred, pos_score = [], [], []
        for rec in recs:
            q = rec["question"]; s = rec["sequences"][0]["sequence"]; m = rec["sequences"][0]["modality"]
            pt = prompt(q, s, m)
            lp = [cand_logprob(model, tok, pt, c, dev) for c in cands]
            lp = np.array(lp)
            k = int(lp.argmax()); pred.append(k)
            if prim == "noul":
                # positive = 'yes' index 0
                pos_score.append(lp[0] - lp[1])
            # gold index
            if prim == "score":
                ys.append(int(rec[gold_key]))
            elif prim == "noul":
                ys.append(0 if rec["answer"] == "yes" else 1)
            else:
                ys.append(cands.index(rec["answer"]) if rec["answer"] in cands else -1)
        ys = np.array(ys); pred = np.array(pred); P = np.array(pos_score) if prim == "noul" else None
        res = {"primitive": prim, "n": len(ys), "accuracy": float((pred == ys).mean())}
        cls = sorted(set(ys.tolist()) | set(pred.tolist())); f1s = []
        for c in cls:
            tp = ((pred == c) & (ys == c)).sum(); fp = ((pred == c) & (ys != c)).sum(); fn = ((pred != c) & (ys == c)).sum()
            pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
            f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
        res["macro_f1"] = float(np.mean(f1s))
        if prim == "noul" and len(set(ys.tolist())) > 1:
            from sklearn.metrics import roc_auc_score
            try: res["auroc"] = float(roc_auc_score(ys, P))
            except Exception: pass
        results[tk] = res
        line = f"  {tk:34s} {prim:6s} n={res['n']:3d} acc={res['accuracy']:.3f}"
        if "auroc" in res: line += f" auroc={res['auroc']:.3f}"
        print(line, flush=True)
    (a.out/"eval_results.json").write_text(json.dumps({"model":"Qwen3-0.6B(frozen)","seq_char_cap":SEQ_CHAR_CAP,
        "max_per_task":a.max_per_task,"results":results}, ensure_ascii=False, indent=2)+"\n")
    print("[#3 DONE]")

if __name__ == "__main__":
    main()

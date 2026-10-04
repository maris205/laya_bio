#!/usr/bin/env python3
"""#4 multimodal baseline: frozen Qwen2.5-VL full-candidate likelihood on images.

For each rendered protein property-map image + question, score every candidate by
the length-normalized log-probability of the candidate text given (image, question),
argmax -> prediction. Frozen VLM, candidate-likelihood (same paradigm as #3 text).
Reports accuracy / macro_f1 for the image version, to compare against the
sequence-version result on the same fold task.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch

JEV = Path("/root/autodl-tmp/jev_gene")
VLM = JEV / "models/Qwen2.5-VL-3B-Instruct"
OUTDIR = JEV / "data/07_multimodal"

@torch.no_grad()
def _cand_lp(model, proc, prompt, img, candidates, dev):
    """length-normalized logprob of each candidate given a prompt (img may be None)."""
    kw = {"images": [img]} if img is not None else {}
    base = proc(text=[prompt], return_tensors="pt", **kw).to(dev)
    base_len = base["input_ids"].shape[1]
    lps = []
    for c in candidates:
        inp = proc(text=[prompt + " " + c], return_tensors="pt", **kw).to(dev)
        ids = inp["input_ids"]
        n_c = ids.shape[1] - base_len
        if n_c <= 0:
            lps.append(float("-inf")); continue
        out = model(**inp)
        logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:]
        tok_lp = logp.gather(1, tgt[:, None]).squeeze(-1)
        lps.append(float(tok_lp[-n_c:].sum().item()) / n_c)
    return np.array(lps)

def score_candidates(model, proc, img, question, candidates, dev):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": question + "\nAnswer with exactly one option."}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return _cand_lp(model, proc, prompt, img, candidates, dev)

def score_candidates_contrastive(model, proc, img, question, candidates, dev):
    """image net contribution: logprob(cand | image+question) - logprob(cand | question only).
    Removes the text-prior dominance that collapses plain candidate-likelihood."""
    q = question + "\nAnswer with exactly one option."
    msgs_img = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    msgs_txt = [{"role": "user", "content": [{"type": "text", "text": q}]}]
    p_img = proc.apply_chat_template(msgs_img, tokenize=False, add_generation_prompt=True)
    p_txt = proc.apply_chat_template(msgs_txt, tokenize=False, add_generation_prompt=True)
    lp_img = _cand_lp(model, proc, p_img, img, candidates, dev)
    lp_txt = _cand_lp(model, proc, p_txt, None, candidates, dev)
    return lp_img - lp_txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=OUTDIR / "fold_image_tasks.jsonl")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--contrastive", action="store_true", help="score = logprob(cand|img+q) - logprob(cand|q)")
    ap.add_argument("--out", type=Path, default=JEV / "artifacts/benchmark_v2_eval/vlm_fold")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(str(VLM))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(VLM), torch_dtype=torch.bfloat16).to(dev).eval()
    items = [json.loads(l) for l in a.manifest.read_text().splitlines() if l.strip()][:a.max]
    ys, preds = [], []
    for i, it in enumerate(items):
        img = it["images"][0]["path"]; cands = it["candidates"]; ans = it["answer"]
        lps = (score_candidates_contrastive if a.contrastive else score_candidates)(model, proc, img, it["question"], cands, dev)
        k = int(lps.argmax()); preds.append(k)
        ys.append(cands.index(ans) if ans in cands else -1)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(items)} running acc={np.mean(np.array(preds)==np.array(ys)):.3f}", flush=True)
    y = np.array(ys); p = np.array(preds)
    acc = float((p == y).mean())
    cls = sorted(set(y.tolist()) | set(p.tolist())); f1s = []
    for c in cls:
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    res = {"model": "Qwen2.5-VL-3B-Instruct(frozen)", "decoding": "contrastive" if a.contrastive else "plain", "task": "fold_image (route A synthetic render)",
           "n": len(y), "accuracy": acc, "macro_f1": float(np.mean(f1s)),
           "label_dist": dict(Counter(items[i]["answer"] for i in range(len(items)))),
           "pred_dist": {cands_name: int((p == j).sum()) for j, cands_name in enumerate(items[0]["candidates"])}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "eval_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

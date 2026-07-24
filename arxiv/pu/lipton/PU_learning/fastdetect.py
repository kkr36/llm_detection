"""Fast-DetectGPT (arXiv:2310.05130) conditional-probability curvature, analytic variant.

Zero-shot AI-text detection baseline. Given a single scoring model p_theta, and taking the
sampling model q_phi = p_theta (the paper's recommended "analytic"/single-model form, which
needs no sampling and only one forward pass per text):

    ll     = log p(x_j | x_<j)                              gathered at the observed token
    mu     = sum_v p(v|x_<j) * log p(v|x_<j)                E_q[log p]
    sigma2 = sum_v p(v|x_<j) * log p(v|x_<j)^2  -  mu^2     Var_q[log p]
    d(x)   = (sum_j ll - sum_j mu) / sqrt(sum_j sigma2)

d(x) is HIGH for machine-generated text and LOW for human text. Equivalent to
`get_sampling_discrepancy_analytic` in the reference implementation
(github.com/baoguangsheng/fast-detect-gpt).

Stage 2 of 3 -- see dump_fastdetect_texts.py for the pipeline overview. Reads the dumped
text JSONs, writes a sha1(text) -> score cache so the 5 seeds (which only reshuffle the same
parquet rows) and repeated runs share work.

Usage (env: /home/kkr36/.conda/envs/llm_embeddings, on a GPU node):
    python fastdetect.py --granularity sentence --model EleutherAI/gpt-neo-2.7B
    python fastdetect.py --granularity sentence --model Qwen/Qwen2.5-7B --batch-size 8
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS_BASE = "/share/garg/arxiv_kaggle/fastdetect_texts"
SCORES_BASE = "/share/garg/arxiv_kaggle/fastdetect_scores"


def model_slug(model_name):
    return model_name.replace("/", "__")


def text_key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class FastDetectScorer:
    """Analytic conditional-probability curvature with a single scoring model."""

    def __init__(self, model_name, device="cuda", dtype=torch.bfloat16, max_length=512,
                 batch_size=16, vocab_chunk=8192):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.vocab_chunk = vocab_chunk

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # right padding: we mask explicitly, and left padding would shift the causal context
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def _score_batch(self, texts):
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_length)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attn).logits

        # predict token t+1 from position t
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask = attn[:, 1:].to(torch.float32)

        # reductions in fp32: bf16 loses too much precision in the sum over ~50k-152k vocab
        log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
        ll = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        # sum_v p*logp and sum_v p*logp^2, chunked over vocab to bound peak memory
        # (B x L x V is 152k-wide for Qwen2.5 and will OOM if materialised twice at once)
        mu = torch.zeros_like(ll)
        second = torch.zeros_like(ll)
        V = log_probs.size(-1)
        for start in range(0, V, self.vocab_chunk):
            lp = log_probs[:, :, start:start + self.vocab_chunk]
            p = lp.exp()
            mu += (p * lp).sum(-1)
            second += (p * lp * lp).sum(-1)
        sigma2 = second - mu * mu

        n_tok = mask.sum(-1)
        ll_sum = (ll * mask).sum(-1)
        mu_sum = (mu * mask).sum(-1)
        var_sum = (sigma2 * mask).sum(-1).clamp_min(0)

        d = (ll_sum - mu_sum) / var_sum.sqrt()
        # a 1-token sequence has no prediction position; sigma2 is degenerate -> NaN
        d = torch.where(n_tok >= 1, d, torch.full_like(d, float("nan")))
        d = torch.where(var_sum > 0, d, torch.full_like(d, float("nan")))
        return d.detach().cpu().numpy().astype(np.float64)

    def score_texts(self, texts, show_progress=True):
        """Scores in input order. Batches length-sorted to cut padding waste."""
        order = np.argsort([len(t) for t in texts])
        out = np.full(len(texts), np.nan, dtype=np.float64)

        rng = range(0, len(order), self.batch_size)
        for i in tqdm(rng, disable=not show_progress, smoothing=.3):
            idx = order[i:i + self.batch_size]
            batch = [texts[j] for j in idx]
            out[idx] = self._score_batch(batch)
        return out


class ScoreCache:
    """sha1(text) -> score, persisted as a single npz per (model, granularity)."""

    def __init__(self, model_name, granularity):
        self.path = os.path.join(SCORES_BASE, model_slug(model_name), f"{granularity}.npz")
        self.keys, self.vals = [], []
        self._map = {}
        if os.path.exists(self.path):
            z = np.load(self.path, allow_pickle=False)
            self.keys = list(z["keys"])
            self.vals = list(z["vals"])
            self._map = {k: v for k, v in zip(self.keys, self.vals)}
            print(f"cache: loaded {len(self._map)} scores from {self.path}")

    def missing(self, texts):
        seen, out = set(), []
        for t in texts:
            k = text_key(t)
            if k not in self._map and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def update(self, texts, scores):
        for t, s in zip(texts, scores):
            self._map[text_key(t)] = float(s)

    def lookup(self, texts):
        return np.array([self._map.get(text_key(t), np.nan) for t in texts], dtype=np.float64)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        keys = np.array(list(self._map.keys()))
        vals = np.array(list(self._map.values()), dtype=np.float64)
        np.savez_compressed(self.path, keys=keys, vals=vals)
        print(f"cache: saved {len(vals)} scores -> {self.path}")


def collect_texts(granularity, llms):
    """Every distinct text referenced by the dumped eval + calib JSONs."""
    root = os.path.join(TEXTS_BASE, granularity)
    texts, seen = [], set()

    for kind in ("eval", "calib"):
        for path in sorted(glob.glob(os.path.join(root, kind, "*", "seed_*.json"))):
            llm = os.path.basename(os.path.dirname(path)).replace("_", " ")
            if llms and llm not in llms:
                continue
            with open(path) as f:
                blob = json.load(f)
            fields = ("p_texts", "u_texts") if kind == "eval" else ("human_texts", "llm_texts")
            for field in fields:
                for t in blob[field]:
                    k = text_key(t)
                    if k not in seen:
                        seen.add(k)
                        texts.append(t)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["sentence", "abstract"], default="sentence")
    ap.add_argument("--model", default="EleutherAI/gpt-neo-2.7B")
    ap.add_argument("--llms", nargs="*", default=None,
                    help="restrict to these test LLMs (default: everything dumped)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--vocab-chunk", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=20000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} granularity={args.granularity}")

    texts = collect_texts(args.granularity, args.llms)
    print(f"{len(texts)} distinct texts referenced")

    cache = ScoreCache(args.model, args.granularity)
    todo = cache.missing(texts)
    print(f"{len(todo)} need scoring ({len(texts)-len(todo)} cache hits)")

    if not todo:
        print("nothing to do")
        return

    scorer = FastDetectScorer(args.model, device=device, max_length=args.max_length,
                              batch_size=args.batch_size, vocab_chunk=args.vocab_chunk)

    for start in range(0, len(todo), args.save_every):
        chunk = todo[start:start + args.save_every]
        scores = scorer.score_texts(chunk)
        cache.update(chunk, scores)
        cache.save()  # crash-safe: scoring is the expensive part
        n_nan = int(np.isnan(scores).sum())
        print(f"chunk {start}-{start+len(chunk)}: {n_nan} NaN "
              f"(mean d = {np.nanmean(scores):.4f})")

    print("done")


if __name__ == "__main__":
    main()

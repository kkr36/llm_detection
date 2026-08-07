"""
Stage 3 -- descriptive stats, interpretability, and the subject-area confounder test.

Confounder question (from the prompt): 2025 abstracts get higher P(LLM) than 2020
human abstracts. Is that partly because 2025 has more deep-learning / ML content and
the detector systematically rates that content as AI-written -- even when genuinely
human? If so, the 2025 rise would be a topic artifact, not a real writing shift.

Three tests:
  (1) On 2020 abstracts (pre-ChatGPT, ground-truth human): does P(LLM) rise with
      deep-learning content / by subject area? If yes -> topic drives false positives.
  (2) Topic-distribution shift: is DL/ML content more prevalent in 2025 than 2020?
  (3) Category-adjusted gap: reweight 2025 to the 2020 subject mix; how much of the
      2020->2025 P(LLM) increase survives?
"""
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis/data"
FIG = "/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis/figures"

C2020, C2025 = "#3b6fb0", "#e08214"          # blue=2020, orange=2025 (colorblind-safe)
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 130})

DL_KEYWORDS = [
    "deep learning", "neural network", "neural net", "transformer", "convolutional",
    "cnn", "rnn", "lstm", "attention mechanism", "self-attention", "gan",
    "generative adversarial", "bert", "gpt", " llm", "large language model",
    "embedding", "fine-tun", "finetun", "pretrain", "pre-train", "self-supervised",
    "reinforcement learning", "gradient descent", "backpropagation", "deep neural",
    "encoder", "decoder", "diffusion model", "foundation model",
]


def dl_count(text):
    t = " " + str(text).lower() + " "
    return sum(t.count(k) for k in DL_KEYWORDS)


COARSE = {
    "cs.LG": "ML/DL", "stat.ML": "ML/DL", "cs.NE": "ML/DL", "cs.AI": "AI",
    "cs.CV": "Vision", "cs.CL": "NLP", "cs.RO": "Robotics",
    "cs.CR": "Security", "cs.IT": "Info/Systems", "cs.SY": "Info/Systems",
    "eess.SY": "Info/Systems", "cs.DC": "Info/Systems", "cs.NI": "Info/Systems",
    "cs.DS": "Theory", "cs.CC": "Theory", "cs.LO": "Theory",
}


def coarse_group(primary):
    p = str(primary)
    if p in COARSE:
        return COARSE[p]
    if p.startswith("cs."):
        return "Other CS"
    if p.startswith("stat"):
        return "ML/DL"
    if p.startswith("eess"):
        return "Info/Systems"
    if p.startswith("math"):
        return "Math"
    return "Other"


# --------------------------------------------------------------- load + enrich
cats = pd.read_parquet(f"{DATA}/categories.parquet")[
    ["prefix_key", "categories", "primary_category", "arxiv_id"]]

abs2020 = pd.read_csv(f"{DATA}/2020_human_sample.csv").merge(cats, on="prefix_key", how="left")
abs2025 = pd.read_csv(f"{DATA}/2025_abstract_sample.csv").merge(cats, on="prefix_key", how="left")

for d in (abs2020, abs2025):
    d["primary_category"] = d["primary_category"].fillna("unknown")
    d["coarse_group"] = d["primary_category"].map(coarse_group)
    d["dl_count"] = d["human_abstract"].map(dl_count)
    d["dl_flag"] = (d["dl_count"] > 0).astype(int)
    d["abstract_len_chars"] = d["human_abstract"].str.len()

abs2020.to_csv(f"{DATA}/2020_human_sample.csv", index=False)     # resave enriched
abs2025.to_csv(f"{DATA}/2025_abstract_sample.csv", index=False)

print(f"[analyze] category match: 2020 {abs2020['arxiv_id'].notna().mean():.0%}  "
      f"2025 {abs2025['arxiv_id'].notna().mean():.0%}")

PLLM = "p_llm_mean_over_models"

# split the single combined sentence file back out by year
sent_all = pd.read_csv(f"{DATA}/sentence_predictions.csv")
sent2020 = sent_all[sent_all["year"] == 2020].reset_index(drop=True)
sent2025 = sent_all[sent_all["year"] == 2025].reset_index(drop=True)

# =============================================================== descriptive
def summ(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=x.mean(), std=x.std(), p10=np.percentile(x, 10),
                median=np.median(x), p90=np.percentile(x, 90),
                frac_gt_0_5=(x > 0.5).mean(), frac_gt_0_9=(x > 0.9).mean())

desc_rows = [
    {"group": "2020 abstract P(LLM) [ground-truth human]", **summ(abs2020[PLLM])},
    {"group": "2025 abstract P(LLM) [unknown]", **summ(abs2025[PLLM])},
    {"group": "2020 sentence P(LLM)", **summ(sent2020["p_llm_mean_over_models"])},
    {"group": "2025 sentence P(LLM)", **summ(sent2025["p_llm_mean_over_models"])},
]
# per-model abstract distribution too (shows agreement across detectors)
for mi in range(5):
    desc_rows.append({"group": f"2020 abstract P(LLM) model{mi}", **summ(abs2020[f'p_llm_m{mi}'])})
    desc_rows.append({"group": f"2025 abstract P(LLM) model{mi}", **summ(abs2025[f'p_llm_m{mi}'])})
desc = pd.DataFrame(desc_rows)
desc.to_csv(f"{DATA}/descriptive_stats.csv", index=False)
print("[analyze] descriptive_stats.csv\n", desc.head(4).to_string(index=False))

# 2020 is genuine human -> any P(LLM) mass above threshold is a false positive
fp_rows = [{"threshold": t,
            "fpr_2020_abstracts": (abs2020[PLLM] > t).mean(),
            "flag_rate_2025_abstracts": (abs2025[PLLM] > t).mean()}
           for t in [0.1, 0.25, 0.5, 0.75, 0.9]]
pd.DataFrame(fp_rows).to_csv(f"{DATA}/threshold_flag_rates.csv", index=False)

# =============================================================== interpretability
allsent = pd.concat([sent2020, sent2025], ignore_index=True)
allsent["sent_len_chars"] = allsent["sentence"].astype(str).str.len()
allsent["sent_len_words"] = allsent["sentence"].astype(str).str.split().map(len)
bins = [0, 60, 100, 140, 180, 220, 1000]
allsent["len_bin"] = pd.cut(allsent["sent_len_chars"], bins)
len_tab = allsent.groupby("len_bin", observed=True)["p_llm_mean_over_models"].agg(["mean", "count"])
len_tab.to_csv(f"{DATA}/interp_sentence_length_vs_pllm.csv")

def tokenize(s):
    return re.findall(r"[a-z]{3,}", str(s).lower())

hi = allsent[allsent["p_llm_mean_over_models"] > 0.7]["sentence"]
lo = allsent[allsent["p_llm_mean_over_models"] < 0.3]["sentence"]
chi, clo = Counter(), Counter()
for s in hi: chi.update(set(tokenize(s)))
for s in lo: clo.update(set(tokenize(s)))
Nhi, Nlo = max(len(hi), 1), max(len(lo), 1)
vocab = {w for w in (set(chi) | set(clo)) if chi[w] + clo[w] >= 8}
lo_rows = []
for w in vocab:
    a, b = chi[w] + 0.5, clo[w] + 0.5
    logodds = np.log((a / (Nhi - a + 1)) / (b / (Nlo - b + 1)))
    lo_rows.append({"word": w, "n_hi": chi[w], "n_lo": clo[w], "log_odds_hi_vs_lo": logodds})
wl = pd.DataFrame(lo_rows).sort_values("log_odds_hi_vs_lo", ascending=False)
wl.to_csv(f"{DATA}/interp_word_logodds.csv", index=False)
print("\n[analyze] words most predictive of HIGH P(LLM):")
print(wl.head(20)[["word", "n_hi", "n_lo", "log_odds_hi_vs_lo"]].to_string(index=False))
print("\n[analyze] words most predictive of LOW P(LLM) (human):")
print(wl.tail(15)[["word", "n_hi", "n_lo", "log_odds_hi_vs_lo"]].to_string(index=False))

allsent.sort_values("p_llm_mean_over_models", ascending=False).head(30)[
    ["year", "abstract_id", "sentence", "p_llm_mean_over_models"]
].to_csv(f"{DATA}/interp_top_llm_sentences.csv", index=False)
allsent.sort_values("p_llm_mean_over_models").head(30)[
    ["year", "abstract_id", "sentence", "p_llm_mean_over_models"]
].to_csv(f"{DATA}/interp_top_human_sentences.csv", index=False)

# =============================================================== CONFOUNDER
dl_effect_2020 = {
    "mean_pllm_2020_DL_abstracts": abs2020.loc[abs2020.dl_flag == 1, PLLM].mean(),
    "mean_pllm_2020_nonDL_abstracts": abs2020.loc[abs2020.dl_flag == 0, PLLM].mean(),
    "n_DL_2020": int((abs2020.dl_flag == 1).sum()),
    "n_nonDL_2020": int((abs2020.dl_flag == 0).sum()),
    "corr_pllm_dlcount_2020": abs2020[PLLM].corr(abs2020["dl_count"]),
}
dl_effect_2025 = {
    "mean_pllm_2025_DL_abstracts": abs2025.loc[abs2025.dl_flag == 1, PLLM].mean(),
    "mean_pllm_2025_nonDL_abstracts": abs2025.loc[abs2025.dl_flag == 0, PLLM].mean(),
    "corr_pllm_dlcount_2025": abs2025[PLLM].corr(abs2025["dl_count"]),
}

def cat_table(col):
    g20 = abs2020.groupby(col).agg(n_2020=(PLLM, "size"), mean_pllm_2020=(PLLM, "mean"))
    g25 = abs2025.groupby(col).agg(n_2025=(PLLM, "size"), mean_pllm_2025=(PLLM, "mean"))
    t = g20.join(g25, how="outer")
    t["share_2020"] = t["n_2020"] / t["n_2020"].sum()
    t["share_2025"] = t["n_2025"] / t["n_2025"].sum()
    return t.reset_index()

cat_coarse = cat_table("coarse_group").sort_values("n_2025", ascending=False)
cat_coarse.to_csv(f"{DATA}/confounder_by_coarse_group.csv", index=False)
cat_primary = cat_table("primary_category").sort_values("n_2025", ascending=False)
cat_primary.to_csv(f"{DATA}/confounder_by_primary_category.csv", index=False)
print("\n[analyze] P(LLM) by coarse subject group:\n", cat_coarse.to_string(index=False))

overall_gap = abs2025[PLLM].mean() - abs2020[PLLM].mean()
w2020 = abs2020["coarse_group"].value_counts(normalize=True)
grp_mean_2025 = abs2025.groupby("coarse_group")[PLLM].mean()
common = [g for g in w2020.index if g in grp_mean_2025.index]
w = w2020.loc[common] / w2020.loc[common].sum()
pllm_2025_adj = float((grp_mean_2025.loc[common] * w).sum())
adjusted_gap = pllm_2025_adj - abs2020[PLLM].mean()
pct_explained = (overall_gap - adjusted_gap) / overall_gap if overall_gap else np.nan

summary = {
    "mean_pllm_2020": abs2020[PLLM].mean(),
    "mean_pllm_2025": abs2025[PLLM].mean(),
    "overall_gap_2025_minus_2020": overall_gap,
    "mean_pllm_2025_reweighted_to_2020_topicmix": pllm_2025_adj,
    "category_adjusted_gap": adjusted_gap,
    "fraction_of_gap_explained_by_subject_shift": pct_explained,
    **dl_effect_2020, **dl_effect_2025,
}
pd.DataFrame([summary]).to_csv(f"{DATA}/confounder_summary.csv", index=False)
print("\n[analyze] CONFOUNDER SUMMARY")
for k, v in summary.items():
    print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

# =============================================================== FIGURES
# Fig 1: abstract-level P(LLM) distributions
fig, ax = plt.subplots(figsize=(7, 4.2))
b = np.linspace(0, 1, 26)
ax.hist(abs2020[PLLM], bins=b, density=True, alpha=0.55, color=C2020,
        label=f"2020 human (mean {abs2020[PLLM].mean():.2f})")
ax.hist(abs2025[PLLM], bins=b, density=True, alpha=0.55, color=C2025,
        label=f"2025 unknown (mean {abs2025[PLLM].mean():.2f})")
ax.set_xlabel("Abstract-level P(LLM), mean over 5 detectors"); ax.set_ylabel("density")
ax.set_title("Detector P(LLM): 2020 (ground-truth human) vs 2025", color=INK)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_pllm_distribution.png"); plt.close(fig)

# Fig 2: mean P(LLM) by coarse subject group
cc = cat_coarse.dropna(subset=["mean_pllm_2020"]).copy()
cc = cc[cc["n_2020"].fillna(0) >= 3]
y = np.arange(len(cc)); h = 0.38
fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(cc) + 1.5))
ax.barh(y + h/2, cc["mean_pllm_2020"], height=h, color=C2020, label="2020 human")
ax.barh(y - h/2, cc["mean_pllm_2025"], height=h, color=C2025, label="2025 unknown")
ax.set_yticks(y); ax.set_yticklabels(cc["coarse_group"])
ax.set_xlabel("mean P(LLM)"); ax.set_title("Mean P(LLM) by subject area", color=INK)
ax.legend(frameon=False); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_pllm_by_subject.png"); plt.close(fig)

# Fig 3: subject-area distribution shift
cc2 = cat_coarse.copy()
cc2 = cc2[(cc2["share_2020"].fillna(0) + cc2["share_2025"].fillna(0)) > 0.01]
y = np.arange(len(cc2))
fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(cc2) + 1.5))
ax.barh(y + h/2, cc2["share_2020"].fillna(0), height=h, color=C2020, label="2020")
ax.barh(y - h/2, cc2["share_2025"].fillna(0), height=h, color=C2025, label="2025")
ax.set_yticks(y); ax.set_yticklabels(cc2["coarse_group"])
ax.set_xlabel("share of sampled abstracts")
ax.set_title("Subject-area mix: 2020 vs 2025", color=INK)
ax.legend(frameon=False); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_subject_shift.png"); plt.close(fig)

# Fig 4: the confounder in one view -- P(LLM) DL vs non-DL, both years
fig, ax = plt.subplots(figsize=(6.4, 4.2))
data = [abs2020.loc[abs2020.dl_flag == 0, PLLM], abs2020.loc[abs2020.dl_flag == 1, PLLM],
        abs2025.loc[abs2025.dl_flag == 0, PLLM], abs2025.loc[abs2025.dl_flag == 1, PLLM]]
bp = ax.boxplot(data, positions=[0, 1, 2.4, 3.4], widths=0.7, patch_artist=True,
                showfliers=False, medianprops=dict(color=INK))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(C2020 if i < 2 else C2025); patch.set_alpha(0.6)
ax.set_xticks([0, 1, 2.4, 3.4])
ax.set_xticklabels(["2020\nnon-DL", "2020\nDL", "2025\nnon-DL", "2025\nDL"])
ax.set_ylabel("abstract P(LLM)")
ax.set_title("Does deep-learning content raise P(LLM) on real 2020 human text?",
             color=INK, fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_dl_confounder.png"); plt.close(fig)

# Fig 5: sentence-level P(LLM) distribution (bimodality check), both years
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(sent2020["p_llm_mean_over_models"], bins=b, density=True, alpha=0.55, color=C2020,
        label=f"2020 sentences (mean {sent2020['p_llm_mean_over_models'].mean():.2f})")
ax.hist(sent2025["p_llm_mean_over_models"], bins=b, density=True, alpha=0.55, color=C2025,
        label=f"2025 sentences (mean {sent2025['p_llm_mean_over_models'].mean():.2f})")
ax.set_xlabel("Sentence-level P(LLM), mean over 5 detectors"); ax.set_ylabel("density")
ax.set_title("Sentence-level P(LLM): 2020 vs 2025", color=INK)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_sentence_distribution.png"); plt.close(fig)

print("\n[analyze] wrote 5 figures to", FIG)
print("[analyze] DONE")

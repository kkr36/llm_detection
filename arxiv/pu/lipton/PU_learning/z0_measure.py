import glob, os, numpy as np, torch, pandas as pd
from sklearn.metrics import roc_auc_score
from model_helper import get_model
from model_inference import get_preds_xy
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEEDS = [0, 1, 2]
def load(gran, seed):
    ep = '3' if gran == 'sentence' else '1'
    net = get_model("DistilBertTTT")
    pt = sorted(glob.glob(f"/share/garg/arxiv_kaggle/ttt_models/PN_TTT_{gran}_{seed}/xy_{ep}/PN_TTT_*.pt"))[-1]
    sd = torch.load(pt, map_location=dev); sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    net.load_state_dict(sd); net.eval(); net.to(dev); return net
rows = []
print("frozen AUC on the v2 parquet (detector trained on rewrite_X):")
for gran, sent in [("sentence", True), ("abstract", False)]:
    for col in ["rewrite_X", "rewrite_Z", "rewrite_strategy_Z_0"]:
        a = []
        for s in SEEDS:
            net = load(gran, s)
            pos, unl, tgt = get_preds_xy(net, dev, 0.5, False, s, sent, True, col)
            a.append(roc_auc_score((np.asarray(tgt) == 1).astype(int), 1 - unl[:, 0]))
        rows.append((gran, col, float(np.mean(a)), float(np.std(a))))
        tag = "  <- in-distribution" if col == "rewrite_X" else ""
        print(f"  {gran:9s} {col:22s} AUC={np.mean(a):.4f}{tag}")
pd.DataFrame(rows, columns=["gran", "col", "frozen_auc", "std"]).to_csv(
    "ttt_logging/csv/z0_shift_severity.csv", index=False)
print("wrote ttt_logging/csv/z0_shift_severity.csv")

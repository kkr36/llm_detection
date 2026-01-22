# for each alpha:
    # load in models, given path (both the pt file with pn and the pt file with pu)
    
    # for each model:
        # given path, parse out what experiments you want to rerun/ assemble test sets:
            # if combine, need to fix 2014-2020 (or whatever interval; most recently 2018-2020)
        # for each test set:
            # get: preds (save these in the same folder with year and alpha)
            # bbe (keep upper/lower conf bounds returned by function), 
            # avg pred pos / neg / avg(avg pos, avg neg) / avg(tpr, fpr) aka plugin (bootstrap 90% bounds?)
            # put into a new csv (train_year, train_method, train_alpha, test_alpha, test_year, **test_metrics); save/add to global df

import os
import pandas as pd
from pathlib import Path
import numpy as np
from model_inference import get_preds
from collections import defaultdict
from model_helper import *

from prepare_metrics import *
from estimator import BBE_estimator

# prior_csv = None # if something, will need to load
# if prior_csv:
#     prior_df = pd.read_csv(prior_csv)
# exit_path = entrance_path if not prior_csv else f"{prior_csv.split('.csv')[0]}1.csv"

def get_metrics(preds_p, preds_u, u_targets):

    preds_up = preds_u[u_targets==0][:,0]
    preds_un = preds_u[u_targets==1][:,0]

    orig_keys = [
        'auc',
        'pos_prob',
        'neg_prob',
        'avg_pos_neg_prob',
        'tpr',
        'fnr',
        'tnr',
        'tpr',
        'plugin',
        'plugin-int' # TODO implement -- average of all predictions, across p/u sets
        'entropy',
        'bbe'
    ]
    bounds = []
    for metric in orig_keys:
        bounds.append(f"{metric}_u")
        bounds.append(f"{metric}_l")
    all_keys = orig_keys + bounds

    metrics_dict = {
        k: None
        for k in all_keys
    }

    # ============================================================================
    # Compute all metrics
    # ============================================================================

    print('calculating metrics')
    metrics_dict['auc'], metrics_dict['auc_l'], metrics_dict['auc_u'] = bootstrap_metric(auc_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['pos_prob'], metrics_dict['pos_prob_l'], metrics_dict['pos_prob_u'] = bootstrap_metric(pos_prob_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['neg_prob'], metrics_dict['neg_prob_l'], metrics_dict['neg_prob_u'] = bootstrap_metric(neg_prob_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['avg_pos_neg_prob'], metrics_dict['avg_pos_neg_prob_l'], metrics_dict['avg_pos_neg_prob_u'] = bootstrap_metric(avg_prob_fn, preds_p, preds_u, n_bootstrap)
    metrics_dict['tpr'], metrics_dict['tpr_l'], metrics_dict['tpr_u'] = bootstrap_metric(tpr_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['fnr'], metrics_dict['fnr_l'], metrics_dict['fnr_u'] = bootstrap_metric(fnr_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['tnr'], metrics_dict['tnr_l'], metrics_dict['tnr_u'] = bootstrap_metric(tnr_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['fpr'], metrics_dict['fpr_l'], metrics_dict['fpr_u'] = bootstrap_metric(fpr_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['plugin'], metrics_dict['plugin_l'], metrics_dict['plugin_u'] = bootstrap_metric(plugin_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['plugin-int'], metrics_dict['plugin-int_l'], metrics_dict['plugin-int_u'] = bootstrap_metric(plugin_int_fn, preds_up, preds_un, n_bootstrap)
    metrics_dict['entropy'], metrics_dict['entropy_l'], metrics_dict['entropy_u'] = bootstrap_metric(entropy_fn, preds_up, preds_up, n_bootstrap)

    # bbe with confidence bounds
    metrics_dict['bbe'], bbe_l, bbe_u = BBE_estimator(preds_p, preds_u, u_targets)
    metrics_dict['bbe_l'], metrics_dict['bbe_u'] = bbe_l[np.argmin(bbe_u)], bbe_u[np.argmin(bbe_u)]

    return metrics_dict

# enter: entrance file, alphas, prior csv (none == make a blank csv, path == append to the existing csv and save to new name? eg have an index 0 that keeps going up each time you pass it in)

entrance_path = "logging_accuracy_temporal_alpha_full_sentence"

alphas = [0, .15, .3, .45, .6]

data_type = "ArXiv_BERT"

flip = False

combine = "combine" in entrance_path

sentence = True # can toggle

clean = True # can toggle

gemini = False

add = False

device = 'cpu' # can toggle

epochs = 3 # can toggle

metrics_dict = defaultdict(list)

for alpha in alphas:
    alpha_dir = Path(f"{entrance_path}/{'sentence' if sentence else 'abstract'}_2020/{alpha}/ArXiv_BERT_{epochs}")
    pts_pu = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "TEDn" in p.name][0]
    pts_pn = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "PN" in p.name][0]
    pts = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt")]
    assert(len(pts) == 2)

    model_paths = {
        "PU": pts_pu,
        "PN": pts_pn 
    }

    for model_name, model_path in model_paths.items():
        # import pdb; pdb.set_trace()
        net = get_model("DistilBert")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)

        None # TODO load this in as pt; use weight dict to load into a seqclassifier obj

        test_alphas = [0.5]
        if alpha not in test_alphas: test_alphas.append(alpha)

        for test_alpha in test_alphas:
            pos_probs, unlabeled_probs, unlabeled_targets = get_preds(data_type, net, device, test_alpha, 2020, combine, sentence, clean, add, gemini, flip)
            info = {
                "learning_method": model_name,
                "data_type": data_type,
                "train_alpha": alpha,
                "test_alpha": test_alpha,
                "epochs": epochs,
                "combine": combine,
                "clean": clean,
                "sentence": sentence,
                "add": add,
                "gemini": gemini,
                "flip": flip,
                "model_path": model_path
            }
            metrics = get_metrics(pos_probs, unlabeled_probs, unlabeled_targets)

            # add metrics and info to metrics_dict
            for a, b in info.items():
                metrics_dict[a] = b
            for a, b in metrics.items():
                metrics_dict[a] = b

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(f"{entrance_path}_alpha.csv")
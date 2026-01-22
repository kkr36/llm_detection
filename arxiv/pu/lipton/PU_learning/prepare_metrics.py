import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Set parameters
n_bootstrap = 1000
alpha = 0.5  # for plugin metric, adjust as needed
threshold = 0.5  # classification threshold

# Generic bootstrap function
def bootstrap_metric(metric_fn, preds_p, preds_u, n_bootstrap=1000, ci=0.90, seed=42):
    """Generic bootstrap function that returns (point_estimate, lower_bound, upper_bound)"""
    # Set seed for reproducibility
    np.random.seed(seed)
    
    estimates = []
    for _ in tqdm(list(range(n_bootstrap)), smoothing=.3):
        estimate = metric_fn(preds_p, preds_u)
        estimates.append(estimate)
    
    point_estimate = metric_fn(preds_p, preds_u)  # Use original data for point estimate
    lower = np.percentile(estimates, (1 - ci) / 2 * 100)
    upper = np.percentile(estimates, (1 + ci) / 2 * 100)
    return point_estimate, lower, upper

# AUC with confidence bounds
def auc_fn(preds_p, preds_u):
    # Resample both positives and negatives
    idx_p = np.random.choice(len(preds_p), len(preds_p), replace=True)
    idx_u = np.random.choice(len(preds_u), len(preds_u), replace=True)
    boot_preds = np.concatenate([preds_p[idx_p], preds_u[idx_u]])
    boot_labels = np.concatenate([np.ones(len(idx_p)), np.zeros(len(idx_u))])
    return roc_auc_score(boot_labels, boot_preds)

# Positive probability (mean prediction for label=1) with confidence bounds
def pos_prob_fn(preds_p, preds_u):
    idx = np.random.choice(len(preds_p), len(preds_p), replace=True)
    return np.mean(preds_p[idx])

# Negative probability (mean prediction for label=0) with confidence bounds
def neg_prob_fn(preds_p, preds_u):
    idx = np.random.choice(len(preds_u), len(preds_u), replace=True)
    return np.mean(preds_u[idx])

# Average of pos and neg prob with confidence bounds
def avg_prob_fn(preds_p, preds_u):
    idx_p = np.random.choice(len(preds_p), len(preds_p), replace=True)
    idx_u = np.random.choice(len(preds_u), len(preds_u), replace=True)
    return (np.mean(preds_p[idx_p]) + np.mean(preds_u[idx_u])) / 2

# TPR (True Positive Rate) with confidence bounds
def tpr_fn(preds_p, preds_u):
    idx = np.random.choice(len(preds_p), len(preds_p), replace=True)
    return np.mean(preds_p[idx] >= threshold)

# FNR (False Negative Rate) with confidence bounds
def fnr_fn(preds_p, preds_u):
    idx = np.random.choice(len(preds_p), len(preds_p), replace=True)
    return np.mean(preds_p[idx] < threshold)

# TNR (True Negative Rate) with confidence bounds
def tnr_fn(preds_p, preds_u):
    idx = np.random.choice(len(preds_u), len(preds_u), replace=True)
    return np.mean(preds_u[idx] < threshold)

# FPR (False Positive Rate) with confidence bounds
def fpr_fn(preds_p, preds_u):
    idx = np.random.choice(len(preds_u), len(preds_u), replace=True)
    return np.mean(preds_u[idx] >= threshold)

# Plugin: alpha * TPR + (1 - alpha) * FPR with confidence bounds
def plugin_fn(preds_p, preds_u):
    idx_p = np.random.choice(len(preds_p), len(preds_p), replace=True)
    idx_u = np.random.choice(len(preds_u), len(preds_u), replace=True)
    boot_tpr = np.mean(preds_p[idx_p] >= threshold)
    boot_fpr = np.mean(preds_u[idx_u] >= threshold)
    return alpha * boot_tpr + (1 - alpha) * boot_fpr

def plugin_int_fn(preds_p, preds_u):
    combined_preds = np.array(preds_p.tolist() + preds_u.tolist())
    idx_c = np.random.choice(len(combined_preds), len(combined_preds), replace=True)
    return np.mean(combined_preds[idx_c])

# Entropy with confidence bounds (assuming binary cross-entropy of predictions)
def entropy_fn(preds_p, preds_u):
    idx_p = np.random.choice(len(preds_p), len(preds_p), replace=True)
    idx_u = np.random.choice(len(preds_u), len(preds_u), replace=True)
    
    # Binary cross-entropy
    eps = 1e-15  # for numerical stability
    boot_preds_p = np.clip(preds_p[idx_p], eps, 1 - eps)
    boot_preds_u = np.clip(preds_u[idx_u], eps, 1 - eps)
    
    entropy_p = -np.mean(np.log(boot_preds_p))
    entropy_u = -np.mean(np.log(1 - boot_preds_u))
    return (entropy_p + entropy_u) / 2
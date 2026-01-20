from matplotlib import pyplot as plt
import pandas as pd
import os
from io import StringIO
import numpy as np
import matplotlib
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18
    }
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator

combine_strs = ['', '_combine']
drop_strs = ['', '_drop']
year = [2010, 2012, 2014, 2016, 2018, 2020][-1]
learning_methods = ['TEDn', 'PN']
data_str = ['sentence', 'abstract'][0]
epochs = 3 if data_str == "sentence" else 2
alphas = [0, .15, .3, .45, .6]

columns = [
    'valyear', 
    'actual_mpe', 
    'valalpha', 
    'our_mpe_estimate', 
    'scott_mpe_estimator', 
    'EN_estimate', 'neg_acc', 'garbage', 'neg_prob', 'pos_acc', 'pos_prob', 'auc'
]

plot_cols = [
    'our_mpe_estimate',
    'neg_acc',
    'neg_prob',
    'pos_acc', # TODO average pos/neg prob to just get plugin
    'pos_prob',
    'auc' # TODO add plots for this
]

def read_csv(path, epochs):
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]  # drop blank lines


    # load as CSV by joining into a CSV-like string

    last4 = lines[epochs+1:]

    df = pd.read_csv(StringIO("\n".join(last4)), header=None)
    df_columns = columns if 'PN' in path else columns[:7] + columns[8:]
    df.columns = df_columns
    return df

if __name__ == "__main__":

    for combine_str in combine_strs:
        aucs_pu_time = []
        aucs_pn_time = []
        tpr_pu_time = []
        tpr_pn_time = []
        tnr_pu_time = []
        tnr_pn_time = []
        pos_pu_time = []
        pos_pn_time = []
        neg_pu_time = []
        neg_pn_time = []
        alpha_diff_pu_time = []
        alpha_diff_pn_time = []
        alpha_0_pu_time = []
        alpha_0_pn_time = []

        aucs_pu_drop = []
        aucs_pn_drop = []
        tpr_pu_drop = []
        tpr_pn_drop = []
        tnr_pu_drop = []
        tnr_pn_drop = []
        pos_pu_drop = []
        pos_pn_drop = []
        neg_pu_drop = []
        neg_pn_drop = []
        alpha_diff_pu_drop = []
        alpha_diff_pn_drop = []
        alpha_0_pu_drop = []
        alpha_0_pn_drop = []

        outer_dir = f"logging_accuracy_temporal_alpha_full_sentence{combine_str}"

        # load all data into csv

        for drop_str in drop_strs:
            year_dir = f"{outer_dir}/{data_str}{drop_str}_{year}"

            for alpha in alphas:
                inner_dir = f"{year_dir}/{alpha}/ArXiv_BERT_3"
                # files_without_period = [
                #     f for f in os.listdir(inner_dir)
                #     if os.path.isfile(os.path.join(inner_dir, f)) and '.' not in f
                # ]
                # import pdb; pdb.set_trace()
                pu_file = [
                    f for f in os.listdir(inner_dir)
                    if 'TEDn' in f
                    and '.pt' not in f
                    and '.pdf' not in f
                ][0]
                pn_file = [
                    f for f in os.listdir(inner_dir)
                    if 'PN' in f
                    and '.pt' not in f
                    and '.pdf' not in f
                ][0]
                pu_file, pn_file = os.path.join(inner_dir, pu_file), os.path.join(inner_dir, pn_file)
                pu_data, pn_data = read_csv(pu_file, epochs), read_csv(pn_file, epochs)
                pu_zero, pn_zero = pu_data[pu_data["valalpha"]==0], pn_data[pn_data["valalpha"]==0]
                pu_alpha, pn_alpha = pu_data[pu_data["valalpha"]==alpha], pn_data[pn_data["valalpha"]==alpha]


                if drop_str == "_drop":
                    aucs_pu_drop.append(pu_zero["auc"])
                    aucs_pn_drop.append(pn_zero["auc"])
                    tpr_pu_drop.append(pu_zero["pos_acc"])
                    tpr_pn_drop.append(pn_zero["pos_acc"])
                    tnr_pu_drop.append(pu_zero["neg_acc"])
                    tnr_pn_drop.append(pn_zero["neg_acc"])
                    pos_pu_drop.append(pu_zero["pos_prob"])
                    pos_pn_drop.append(pn_zero["pos_prob"])
                    neg_pu_drop.append(pu_zero["neg_prob"])
                    neg_pn_drop.append(pn_zero["neg_prob"])
                    alpha_diff_pu_drop.append((pu_alpha["our_mpe_estimate"] - pn_alpha["actual_mpe"]).tolist()[0])
                    alpha_diff_pn_drop.append((pn_alpha["our_mpe_estimate"] - pn_alpha["actual_mpe"]).tolist()[0])
                    alpha_0_pu_drop.append((pu_zero["our_mpe_estimate"] - pu_zero["actual_mpe"]).tolist()[0])
                    alpha_0_pn_drop.append((pn_zero["our_mpe_estimate"] - pn_zero["actual_mpe"]).tolist()[0])

                    # import pdb; pdb.set_trace()
                elif drop_str == "":
                    aucs_pu_time.append(pu_zero["auc"])
                    aucs_pn_time.append(pn_zero["auc"])
                    tpr_pu_time.append(pu_zero["pos_acc"])
                    tpr_pn_time.append(pn_zero["pos_acc"])
                    tnr_pu_time.append(pu_zero["neg_acc"])
                    tnr_pn_time.append(pn_zero["neg_acc"])
                    pos_pu_time.append(pu_zero["pos_prob"])
                    pos_pn_time.append(pn_zero["pos_prob"])
                    neg_pu_time.append(pu_zero["neg_prob"])
                    neg_pn_time.append(pn_zero["neg_prob"])
                    alpha_diff_pu_time.append((pu_alpha["our_mpe_estimate"] - pu_alpha["actual_mpe"]).tolist()[0])
                    alpha_diff_pn_time.append((pn_alpha["our_mpe_estimate"] - pn_alpha["actual_mpe"]).tolist()[0])
                    alpha_0_pu_time.append((pu_zero["our_mpe_estimate"] - pu_zero["actual_mpe"]).tolist()[0])
                    alpha_0_pn_time.append((pn_zero["our_mpe_estimate"] - pn_zero["actual_mpe"]).tolist()[0])
                    # import pdb; pdb.set_trace()

                                

        # ---------- AUC ----------
        auc_data = {
            "PU retrain": aucs_pu_time,
            "PN retrain": aucs_pn_time,
            "PU drop": aucs_pu_drop,
            "PN drop": aucs_pn_drop,
        }

        # ---------- TPR ----------
        tpr_data = {
            "PU retrain": tpr_pu_time,
            "PN retrain": tpr_pn_time,
            "PU drop": tpr_pu_drop,
            "PN drop": tpr_pn_drop,
        }

        # ---------- TNR ----------
        tnr_data = {
            "PU retrain": tnr_pu_time,
            "PN retrain": tnr_pn_time,
            "PU drop": tnr_pu_drop,
            "PN drop": tnr_pn_drop,
        }

        # ---------- POS ----------
        pos_data = {
            "PU retrain": pos_pu_time,
            "PN retrain": pos_pn_time,
            "PU drop": pos_pu_drop,
            "PN drop": pos_pn_drop,
        }

        # ---------- NEG ----------
        neg_data = {
            "PU retrain": neg_pu_time,
            "PN retrain": neg_pn_time,
            "PU drop": neg_pu_drop,
            "PN drop": neg_pn_drop,
        }

        naive_mpe_data = {
            "PU retrain": [(p+n)/2 for n, p in zip(neg_pu_time, pos_pu_time)],
            "PN retrain": [(p+n)/2 for n, p in zip(neg_pn_time, pos_pn_time)],
            "PU no_drop": [(p+n)/2 for n, p in zip(neg_pu_drop, pos_pu_drop)],
            "PN no_drop": [(p+n)/2 for n, p in zip(neg_pn_drop, pos_pn_drop)],
        }

        alpha_diff_data = {
            "PU retrain": alpha_diff_pu_time,
            "PN retrain": alpha_diff_pn_time,
            "PU drop": alpha_diff_pu_drop,
            "PN drop": alpha_diff_pn_drop,
        }

        alpha_0_data = {
            "PU retrain": alpha_0_pu_time,
            "PN retrain": alpha_0_pn_time,
            "PU drop": alpha_0_pu_drop,
            "PN drop": alpha_0_pn_drop,
        }

        # color by experiment type
        colors = {
            "2010": "black",      # matplotlib auto-assigns
            "retrain": "red",
            "drop": "blue",
        }

        def plot_group(data, title, ylabel):
            # ax = plt.gca()
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.figure()

            for label, values in data.items():
                is_pn = label.startswith("PN")
                linestyle = ":" if is_pn else "-"

                if "2010" in label:
                    color = colors["2010"]
                elif "retrain" in label:
                    color = colors["retrain"]
                elif "drop" in label:
                    color = colors["drop"]
                else:
                    color = "red"

                plt.plot(
                    alphas,
                    values,
                    linestyle=linestyle,
                    label=label,
                    color=color
                )

            # plt.xticks(year)
            # plt.xlabel("Alpha")
            # plt.ylabel(ylabel)
            # plt.title(title)
            plt.legend()
            plt.tight_layout()
            if not os.path.exists("alpha"):
                os.makedirs("alpha")
            plt.savefig(f"alpha/{title.replace(' ', '_')}_alpha.pdf", format="pdf", bbox_inches="tight")


        plot_group(auc_data, f"AUC over Time{' Combine' if combine_str else ' 2020'}", "AUC")
        plot_group(tpr_data, f"TPR over Time{' Combine' if combine_str else ' 2020'}", "TPR")
        plot_group(tnr_data, f"TNR over Time{' Combine' if combine_str else ' 2020'}", "TNR")

        plot_group(pos_data, f"Avg Pos over Time{' Combine' if combine_str else ' 2020'}", "Pos")
        plot_group(neg_data, f"Avg Neg over Time{' Combine' if combine_str else ' 2020'}", "Neg")
        plot_group(naive_mpe_data, f"Avg Plug-In over Time{' Combine' if combine_str else ' 2020'}", "Plug In")

        plot_group(alpha_diff_data, f"Avg Alpha Diff over Time{' Combine' if combine_str else ' 2020'}", "Alpha Diff")
        plot_group(alpha_0_data, f"Avg Alpha 0 over Time{' Combine' if combine_str else ' 2020'}", "Alpha 0")
from matplotlib import pyplot as plt
import matplotlib
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18
    }
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator


aucs_pu_time = [.9330083007008012, .9240098426183836, .916448474011335, .911637893053959, .8961374315666221]
aucs_pn_time = [.9331595342460725, .9203580713078868, .9081108576867911, .885986844264572, .873603991892459]
aucs_pu_no_drop = [.9330081234525027, .929788679077837, .9274816233751959, .9250517213590741, .919287413148079]
aucs_pn_no_drop = [.9331595342460725, .9327712799977719, .9290637549294126, .924109936973561, .9225173182283557]

tpr_pu_time = [.8880826230515242, .9145358212393457, .9024034400675728, .8528756814866006, .8597481379098518]
tpr_pn_time = [.887967442217615, .8988328342163864, .8693081471243185, .8726483913076865, .782768947247178]
tpr_pu_no_drop = [.8880826230515242, .8663518390539814, .8928050372418029, .8687322429547724, .8660446901635568]
tpr_pn_no_drop = [.887967442217615, .9113875451124933, .9123089917837672, .9490132841895109, .9417952852645319]

tnr_pu_time = [.8061024144335367, .7322897320244096, .7332448925444415, .8092862828336429, .7617405147253914]
tnr_pn_time = [.8068453170602282, .7690634120456354, .7923056513664102, .7523481029450783, .816609180153887]
tnr_pu_no_drop = [.8061024144335367, .8298222340143274, .7828071106394269, .8149111170071637, .804669673653489]
tnr_pn_no_drop = [.8068453170602282, .7648182541788273, .7524542318917484, .6310957813743698, .6486601220482887]

pos_pu_time = [.8787328004837036, .9097070097923279, .8977020382881165, .8574415445327759, .863559901714325]
pos_pn_time = [.865429699420929, .8042408227920532, .7107706069946289, .6649144887924194, .5653067231178284]
pos_pu_no_drop = [.8787328004837036, .8543462157249451, .8818190693855286, .8571698665618896, .8591654300689697]
pos_pn_no_drop = [.865429699420929, .8896127939224243, .8971531391143799, .9290396571159363, .9231055974960327]

neg_pu_time = [.23741474747657776, .3043118417263031, .31729158759117126, .2343674600124359, .28353574872016907]
neg_pn_time = [.21618875861167908, .25134971737861633, .21902459859848022, .25237786769866943, .21804280579090118]
neg_pu_no_drop = [.23741480708122253, .21727290749549866, .26527050137519836, .2592415511608124, .24519310891628265]
neg_pn_no_drop = [.21618875861167908, .25719961524009705, .2692376971244812, .37841495871543884, .36683204770088196]

alphas = [0, .15, .3, .45, .6]

# # ------------------ Grouped data ------------------
# groups = [
#     ("AUC", "AUC", {
#         "PU time": aucs_pu_time,
#         "PN time": aucs_pn_time,
#         "PU no_drop": aucs_pu_no_drop,
#         "PN no_drop": aucs_pn_no_drop,
#     }),
#     # ("TPR", "TPR", {
#     #     "PU time": tpr_pu_time,
#     #     "PN time": tpr_pn_time,
#     #     "PU no_drop": tpr_pu_no_drop,
#     #     "PN no_drop": tpr_pn_no_drop,
#     # }),
#     # ("TNR", "TNR", {
#     #     "PU time": tnr_pu_time,
#     #     "PN time": tnr_pn_time,
#     #     "PU no_drop": tnr_pu_no_drop,
#     #     "PN no_drop": tnr_pn_no_drop,
#     # }),
#     ("Avg Pos", "Pos", {
#         "PU time": pos_pu_time,
#         "PN time": pos_pn_time,
#         "PU no_drop": pos_pu_no_drop,
#         "PN no_drop": pos_pn_no_drop,
#     }),
#     ("Avg Neg", "Neg", {
#         "PU time": neg_pu_time,
#         "PN time": neg_pn_time,
#         "PU no_drop": neg_pu_no_drop,
#         "PN no_drop": neg_pn_no_drop,
#     }),
# ]

# colors = {
#     "time": "red",
#     "no_drop": "blue",
# }

# # ------------------ Plot ------------------
# fig, axes = plt.subplots(1, 3, figsize=(35, 10), sharex=True)
# axes = axes.flatten()

# legend_handles, legend_labels = None, None

# for i, (title, ylabel, data) in enumerate(groups):
#     ax = axes[i]

#     for label, values in data.items():
#         linestyle = ":" if label.startswith("PN") else "-"
#         color = colors["no_drop"] if "no_drop" in label else colors["time"]

#         line, = ax.plot(
#             alphas,
#             values,
#             linestyle=linestyle,
#             color=color,
#             label=label
#         )

#     ax.set_title(f"({chr(97+i)}) {title}")
#     ax.set_ylabel(ylabel)
#     ax.set_xlabel("Alpha")

#     if legend_handles is None:
#         legend_handles, legend_labels = ax.get_legend_handles_labels()

# # Remove empty 6th panel if only 5 plots
# # fig.delaxes(axes[-1])

# # ------------------ Shared legend ------------------
# # fig.legend(
# #     legend_handles,
# #     legend_labels,
# #     loc="lower center",
# #     ncol=4,
# #     frameon=False,
# #     bbox_to_anchor=(0.5, -0.02)
# # )

# fig.tight_layout(rect=[0, 0.08, 1, 1])
# plt.savefig("six_panel_metrics_alpha.pdf", bbox_inches="tight")
# plt.close()

# ---------- AUC ----------
auc_data = {
    "PU retrain": aucs_pu_time,
    "PN retrain": aucs_pn_time,
    "PU no_drop": aucs_pu_no_drop,
    "PN no_drop": aucs_pn_no_drop,
}

# ---------- TPR ----------
tpr_data = {
    "PU retrain": tpr_pu_time,
    "PN retrain": tpr_pn_time,
    "PU no_drop": tpr_pu_no_drop,
    "PN no_drop": tpr_pn_no_drop,
}

# ---------- TNR ----------
tnr_data = {
    "PU retrain": tnr_pu_time,
    "PN retrain": tnr_pn_time,
    "PU no_drop": tnr_pu_no_drop,
    "PN no_drop": tnr_pn_no_drop,
}

# ---------- POS ----------
pos_data = {
    "PU retrain": pos_pu_time,
    "PN retrain": pos_pn_time,
    "PU no_drop": pos_pu_no_drop,
    "PN no_drop": pos_pn_no_drop,
}

# ---------- NEG ----------
neg_data = {
    "PU retrain": neg_pu_time,
    "PN retrain": neg_pn_time,
    "PU no_drop": neg_pu_no_drop,
    "PN no_drop": neg_pn_no_drop,
}

# color by experiment type
colors = {
    "2010": "black",      # matplotlib auto-assigns
    "retrain": "red",
    "no_drop": "blue",
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
        elif "no_drop" in label:
            color = colors["no_drop"]
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
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(" ", "_")}_alpha.pdf", format="pdf", bbox_inches="tight")


plot_group(auc_data, "AUC over Time", "AUC")
plot_group(tpr_data, "TPR over Time", "TPR")
plot_group(tnr_data, "TNR over Time", "TNR")

plot_group(pos_data, "Avg Pos over Time", "Pos")
plot_group(neg_data, "Avg Neg over Time", "Neg")

# plt.plot(year, aucs_pu_2010, label="PU")
# plt.plot(year, aucs_pn_2010, label="PN")
# plt.plot(year, aucs_pu_time, label="PU, Retrain Per-Year")
# plt.plot(year, aucs_pn_time, label="PN, Retrain Per-Year")
# plt.legend()
# plt.ylabel("AUC")
# plt.xlabel("Year")
# plt.savefig("train_per_year.pdf", format="pdf")
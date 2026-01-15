from matplotlib import pyplot as plt
import matplotlib
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18
    }
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator

aucs_pu_2010 = [.949417631561179, .9416870535850799, .9370601221730392, .9268074390792984, .9270589591237954, .9181899731778503]
aucs_pn_2010 = [.9499054018394624, .9432432338291997, .9382209459137514, .9268141757409668, .9271342395576523, .9205726925118727]
aucs_pu_time = [.949417631561179, .9438511617247468, .9331875864334792, .9233879359447388, .9150915870524778, .8961374315666221]
aucs_pn_time = [.9499054018394624, .9444888534247137, .9209205223497503, .9026675322733436, .8913680158642918, .873603991892459]
aucs_pu_no_drop = [.949417631561179, .9438511617247468, .9366680769027116, .9286355538404001, .9264792685527923, .919287413148079]
aucs_pn_no_drop = [.9499054018394624, .9444888534247137, .9385325531087622, .928947579149108, .9268233814442802, .9225173182283557]

tpr_pu_2010 = [.9136216859279401, .9162239649211569, .9134055295758471, .9155424509249536, .9227469500196773, .9261690854641788]
tpr_pn_2010 = [.9191026512576479, .9209461168732608, .9191894330954139, .9205913240164795, .9279417552144825, .9306611379866391]
tpr_pu_time = [.9136216859279401, .9171093684121764, .8764049552875544, .8617416592616528, .8541912632821723, .8597481379098518]
tpr_pn_time = [.9191026512576479, .9142423475841134, .9053654934777258, .9094838032151223, .870680834317198, .782768947247178]
tpr_pu_no_drop = [.9136216859279401, .9171093684121764, .8839527442776274, .8995072299862671, .8974813065722157, .8660446901635568]
tpr_pn_no_drop = [.9191026512576479, .9142423475841134, .9013454754286652, .9331125292834639, .9489177489177489, .9417952852645319]

tnr_pu_2010 = [.8184436196083451, .78767080158680, .7760029264723814, .7366941753378783, .7366941753378783, .6849031573361635]
tnr_pn_2010 = [.8107641110968898, .7789182041433159, .7662480185343251, .7230594426504007, .7091762601989494, .681560095516052]
tnr_pu_time = [.8184436196083451, .7932120143567786, .8284965248140471, .8196387991867001, .8085950597965799, .7617405147253914]
tnr_pn_time = [.8107641110968898, .8007052452616334, .7742958175832215, .728740581270183, .7685257628255281, .816609180153887]
tnr_pu_no_drop = [.8184436196083451, .7932120143567786, .8238019753688575, .6969860064585576, .7719347267240416, .804669673653489]
tnr_pn_no_drop = [.8107641110968898, .8007052452616334, .8037434459212291, .7707212055974165, .6466413322901531, .6486601220482887]

pos_pu_2010 = [.9040080904960632, .9061768651008606, .9042473435401917, .9059719443321228, .9130527973175049, .9158026576042175]
pos_pn_2010 = [.9033348560333252, .9054661989212036, .90302574634552, .9038094282150269, .9106138348579407, .9141987562179565]
pos_pu_time = [.9040080904960632, .9063254594802856, .8725089430809021, .8624376058578491, .8575981855392456, .863559901714325]
pos_pn_time = [.9033348560333252, .8959008455276489, .8063069581985474, .7191134691238403, .6805798411369324, .5653067231178284]
pos_pu_no_drop = [.9040080904960632, .9142690896987915, .8734449148178101, .8854063749313354, .8900303244590759, .8591654300689697]
pos_pn_no_drop = [.9033348560333252, .8836954236030579, .8852135539054871, .91741544008255, .9330000281333923, .9231055974960327]

neg_pu_2010 = [.2264431267976761, .25512295961380005, .265371173620224, .3040977418422699, .31853967905044556, .3497863709926605]
neg_pn_2010 = [.22003695368766785, .24843336641788483, .25954341888427734, .2996314465999603, .3117976188659668, .3369392454624176]
neg_pu_time = [.2264431267976761, .25199154019355774, .21646322309970856, .22444283962249756, .2396114617586136, .28353574872016907]
neg_pn_time = [.22003695368766785, .2297695130109787, .24583032727241516, .28183239698410034, .2433832287788391, .21804280579090118]
neg_pu_no_drop = [.2264431267976761, .27184489369392395, .22055162489414215, .273965448141098, .2870192229747772, .24519310891628265]
neg_pn_no_drop = [.22003695368766785, .2200368493795395, .23186089098453522, .3208804428577423, .37618353962898254, .36683204770088196]

year = [2010, 2012, 2014, 2016, 2018, 2020]

# # ------------------ Grouped data ------------------
# groups = [
#     ("AUC", "AUC", {
#         "PU 2010": aucs_pu_2010,
#         "PN 2010": aucs_pn_2010,
#         "PU retrain": aucs_pu_time,
#         "PN retrain": aucs_pn_time,
#         "PU no_drop": aucs_pu_no_drop,
#         "PN no_drop": aucs_pn_no_drop,
#     }),
#     ("TPR", "TPR", {
#         "PU time": tpr_pu_time,
#         "PN time": tpr_pn_time,
#         "PU no_drop": tpr_pu_no_drop,
#         "PN no_drop": tpr_pn_no_drop,
#     }),
#     ("TNR", "TNR", {
#         "PU time": tnr_pu_time,
#         "PN time": tnr_pn_time,
#         "PU no_drop": tnr_pu_no_drop,
#         "PN no_drop": tnr_pn_no_drop,
#     }),
#     ("Avg Pos", "Pos", {
#         "PU 2010": pos_pu_2010,
#         "PN 2010": pos_pn_2010,
#         "PU retrain": pos_pu_time,
#         "PN retrain": pos_pn_time,
#         "PU no_drop": pos_pu_no_drop,
#         "PN no_drop": pos_pn_no_drop,
#     }),
#     ("Avg Neg", "Neg", {
#         "PU 2010": neg_pu_2010,
#         "PN 2010": neg_pn_2010,
#         "PU retrain": neg_pu_time,
#         "PN retrain": neg_pn_time,
#         "PU no_drop": neg_pu_no_drop,
#         "PN no_drop": neg_pn_no_drop,
#     }),
# ]

# colors = {
#     "retrain": "red",
#     "no_drop": "blue",
#     "2010": "black"
# }

# # ------------------ Plot ------------------
# fig, axes = plt.subplots(1, 3, figsize=(35, 10), sharex=True)
# axes = axes.flatten()

# legend_handles, legend_labels = None, None

# for i, (title, ylabel, data) in enumerate(groups):
#     ax = axes[i]

#     for label, values in data.items():
#         linestyle = ":" if label.startswith("PN") else "-"
#         color = colors["no_drop"] if "no_drop" in label else colors["retrain"] if "retrain" in label else colors["2010"]

#         line, = ax.plot(
#             year,
#             values,
#             linestyle=linestyle,
#             color=color,
#             label=label
#         )

#     ax.set_title(f"({chr(97+i)}) {title}")
#     ax.set_ylabel(ylabel)
#     ax.set_xlabel("Year")

#     if legend_handles is None:
#         legend_handles, legend_labels = ax.get_legend_handles_labels()

# # Remove empty 6th panel if only 5 plots
# # fig.delaxes(axes[-1])

# # ------------------ Shared legend ------------------
# fig.legend(
#     legend_handles,
#     legend_labels,
#     loc="lower center",
#     ncol=4,
#     frameon=False,
#     bbox_to_anchor=(0.5, -0.02)
# )

# fig.tight_layout(rect=[0, 0.08, 1, 1])
# plt.savefig("six_panel_metrics_temporal.pdf", bbox_inches="tight")
# plt.close()

# ---------- AUC ----------
auc_data = {
    "PU 2010": aucs_pu_2010,
    "PN 2010": aucs_pn_2010,
    "PU retrain": aucs_pu_time,
    "PN retrain": aucs_pn_time,
    "PU no_drop": aucs_pu_no_drop,
    "PN no_drop": aucs_pn_no_drop,
}

# ---------- TPR ----------
tpr_data = {
    "PU 2010": tpr_pu_2010,
    "PN 2010": tpr_pn_2010,
    "PU retrain": tpr_pu_time,
    "PN retrain": tpr_pn_time,
    "PU no_drop": tpr_pu_no_drop,
    "PN no_drop": tpr_pn_no_drop,
}

# ---------- TNR ----------
tnr_data = {
    "PU 2010": tnr_pu_2010,
    "PN 2010": tnr_pn_2010,
    "PU retrain": tnr_pu_time,
    "PN retrain": tnr_pn_time,
    "PU no_drop": tnr_pu_no_drop,
    "PN no_drop": tnr_pn_no_drop,
}

# ---------- POS ----------
pos_data = {
    "PU 2010": pos_pu_2010,
    "PN 2010": pos_pn_2010,
    "PU retrain": pos_pu_time,
    "PN retrain": pos_pn_time,
    "PU no_drop": pos_pu_no_drop,
    "PN no_drop": pos_pn_no_drop,
}

# ---------- NEG ----------
neg_data = {
    "PU 2010": neg_pu_2010,
    "PN 2010": neg_pn_2010,
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
    # Method 1: Using MaxNLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.figure()

    for label, values in data.items():
        is_pn = label.startswith("PN")
        linestyle = ":" if is_pn else "-"

        if "2010" in label:
            color = colors["2010"]
        elif "retrain" in label:
            color = colors["retrain"]
        else:
            color = colors["no_drop"]

        plt.plot(
            year,
            values,
            linestyle=linestyle,
            label=label,
            color=color
        )

    plt.xticks(year)
    # plt.xlabel("Year")
    # plt.ylabel(ylabel)
    # plt.title(title)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(" ", "_")}.pdf", format="pdf", bbox_inches="tight")


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
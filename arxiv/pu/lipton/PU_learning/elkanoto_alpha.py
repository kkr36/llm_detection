from estimator import BBE_estimator2
print("loaded big lib")
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

if __name__ == "__main__":
    alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    output_folder = "../figs"

    preds_path = "/share/garg/arxiv_kaggle/pu/preds.pkl"

    with open(preds_path, 'rb') as f:
        data = pickle.load(f)
    years = sorted(list(data.keys()))
    
    # import pdb; pdb.set_trace()

    train_preds = [data[year]['train'][0][0] for year in data.keys()]
    train_labels = [data[year]['train'][1] for year in data.keys()]

    cal_preds = [data[year]['cal'][0] for year in data.keys()]
    cal_labels = [data[year]['cal'][1] for year in data.keys()]

    # for train/cal, plot curve of pred % llm (population, using lipton) over time
    pred_alphas = {
        'train': [],
        'cal': [],
        'test': {
            x: [] for x in alphas
        }
    }

    assert(len(train_preds) == len(cal_preds))
    for i, year in tqdm(list(enumerate(years))):
        train_preds_year = train_preds[i]
        train_labels_year = train_labels[i]
        cal_preds_year = cal_preds[i]
        cal_preds_labels = cal_labels[i]

        all_pos = None

        for name, preds, labels in [("train", train_preds_year, train_labels_year), ("cal", cal_preds_year, cal_preds_labels)]:
            pos_mask = (labels == 1)
            neg_mask = (labels != 1)
            pos_preds = preds[pos_mask]
            neg_preds = preds[neg_mask]
            pred_alpha = BBE_estimator2(pos_preds, neg_preds)[0]
            # import pdb; pdb.set_trace()
            pred_alphas[name].append(pred_alpha)
            if all_pos is None:
                all_pos = pos_preds
            else:
                all_pos = np.hstack([all_pos, pos_preds])

        # for test, plot heat map of pred % llm (population, using lipton) over alpha and time
        for alpha in alphas:
            test_preds = [data[year]['test'][alpha][0] for year in data.keys()]
            test_labels = [[data[year]['test'][alpha][1]] for year in data.keys()]
            test_preds, test_labels = test_preds[i], test_labels[i]
            # test_preds = test_preds[:,1]
            # pos_mask = (test_labels == 1)
            # neg_mask = (test_labels != 1)
            # import pdb; pdb.set_trace()
            # pos_preds = test_preds[pos_mask]
            # neg_preds = test_preds[neg_mask]
            # print(alpha, len(pos_preds) / len(pos_mask))

            pred_alpha = BBE_estimator2(all_pos, test_preds)[0]
            pred_alphas['test'][alpha].append(pred_alpha)

        with open('/share/garg/arxiv_kaggle/pu/alphas.pkl', 'wb') as f:
            pickle.dump(pred_alphas, f)

    # import pdb; pdb.set_trace()
    # for train/cal sets, plot alpha over time
    def plot_alpha(figname, alphas):
        plt.plot(years, alphas)
        plt.xlabel("Year")
        plt.ylabel("% Pred LLM")
        plt.savefig(figname, format='pdf')
        plt.clf()
    
    plot_alpha("../figs/train_alpha.pdf", pred_alphas['train'])
    plot_alpha("../figs/cal_alpha.pdf", pred_alphas['cal'])

    fig = plt.figure()
    ax = plt.subplot()
    for alpha in alphas:
        ax.plot(years, pred_alphas['test'][alpha], label=alpha)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=6, title='alpha')
    plt.tight_layout()
    plt.xlabel("Year")
    plt.ylabel("% Pred LLM")
    plt.savefig("../figs/test_alpha.pdf", bbox_inches="tight", format='pdf')
    plt.clf()


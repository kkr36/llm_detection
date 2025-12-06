from matplotlib import pyplot as plt
from xgb_yearly_pipeline import nested_dict
import numpy as np
import pickle
plt.rcParams.update({'font.size': 12})

if __name__ == "__main__":
    with open('xgb_metrics.pkl', 'rb') as f:
        data = pickle.load(f)

    # import pdb; pdb.set_trace()

    years = list(range(2010,2026))
    alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.5]

    test_accs = np.array([[data[year]['test_acc'][alpha] for year in years] for alpha in alphas])
    test_probs = np.array([[data[year]['test_prob'][alpha] for year in years] for alpha in alphas])
    test_stds_neg = np.array([[data[year]['test_std_neg'][alpha] for year in years] for alpha in alphas])
    test_stds_pos = np.array([[data[year]['test_std_pos'][alpha] for year in years] for alpha in alphas])
    test_entropy_neg = np.array([[data[year]['test_entropy_neg'][alpha] for year in years] for alpha in alphas])
    test_entropy_pos = np.array([[data[year]['test_entropy_pos'][alpha] for year in years] for alpha in alphas])

    ### HEATMAP PROBS ###
    fig, ax = plt.subplots(figsize=(25, 10))
    im = ax.imshow(test_probs, aspect='auto', cmap='viridis', origin='lower')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Test Probability')

    # Set axis labels
    plt.xlabel('Year')
    plt.ylabel('Alpha')

    # Set tick labels
    plt.xticks(range(0, len(years), 5), [years[i] for i in range(0, len(years), 5)])
    plt.yticks(range(len(alphas)), alphas)

    # Add text annotations to each cell
    for i in range(len(alphas)):
        for j in range(len(years)):
            text = ax.text(j, i, f'{test_probs[i, j]:.3f}',
                        ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig("metrics/test_probs.pdf", format='pdf')
    plt.clf()

    ### PROBS LINE PLOT ###
    fig = plt.figure()
    ax = plt.subplot()
    for (i, alpha) in enumerate(alphas):
        ax.plot(years, test_probs[i], label=alpha)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=6, title='alpha')
    plt.tight_layout()
    plt.xlabel("Year")
    plt.ylabel("% Pred LLM")
    plt.savefig("metrics/test_alpha_plot.pdf", bbox_inches="tight", format='pdf')
    plt.clf()

    ### HEATMAP ACCS ###

    fig, ax = plt.subplots(figsize=(25, 10))
    im = ax.imshow(test_accs, aspect='auto', cmap='viridis', origin='lower')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Test Accuracy')

    # Set axis labels
    plt.xlabel('Year')
    plt.ylabel('Alpha')

    # Set tick labels
    plt.xticks(range(0, len(years), 5), [years[i] for i in range(0, len(years), 5)])
    plt.yticks(range(len(alphas)), alphas)

    # Add text annotations to each cell
    for i in range(len(alphas)):
        for j in range(len(years)):
            text = ax.text(j, i, f'{test_accs[i, j]:.3f}',
                        ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig("metrics/test_accs.pdf", format='pdf')
    plt.clf()

    ### HEATMAP STDs ###

    for sign, arr in [('pos', test_stds_pos), ('neg', test_stds_neg)]:
        fig, ax = plt.subplots(figsize=(25, 10))
        im = ax.imshow(arr, aspect='auto', cmap='viridis', origin='lower')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'Test STD {sign}')

        # Set axis labels
        plt.xlabel('Year')
        plt.ylabel('Alpha')

        # Set tick labels
        plt.xticks(range(0, len(years), 5), [years[i] for i in range(0, len(years), 5)])
        plt.yticks(range(len(alphas)), alphas)

        # Add text annotations to each cell
        for i in range(len(alphas)):
            for j in range(len(years)):
                text = ax.text(j, i, f'{arr[i, j]:.3f}',
                            ha="center", va="center", color="white")

        plt.tight_layout()
        plt.savefig(f"metrics/test_stds_{sign}.pdf", format='pdf')
        plt.clf()
    
    ### HEATMAP ENTROPY ###
    for sign, arr in [('pos', test_entropy_pos), ('neg', test_entropy_neg)]:
        import pdb; pdb.set_trace()
        if sign == 'pos':
            arr = arr[1:,:]

        fig, ax = plt.subplots(figsize=(25, 10))
        im = ax.imshow(arr, aspect='auto', cmap='viridis', origin='lower')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'Test STD {sign}')

        # Set axis labels
        plt.xlabel('Year')
        plt.ylabel('Alpha')

        # Set tick labels
        plt.xticks(range(0, len(years), 5), [years[i] for i in range(0, len(years), 5)])
        if sign == 'pos':
            plt.yticks(range(len(alphas)-1), alphas[1:])
        else:
            plt.yticks(range(len(alphas)), alphas)

        # Add text annotations to each cell
        for i in range(len(alphas) if sign == 'neg' else len(alphas) - 1):
            for j in range(len(years)):
                text = ax.text(j, i, f'{arr[i, j]:.3f}',
                            ha="center", va="center", color="white")

        plt.tight_layout()
        plt.savefig(f"metrics/test_entropy_{sign}.pdf", format='pdf')
        plt.clf()

    ### CAL TIMELINE ###

    plt.subplots(figsize=(12,10))
    plt.plot(years, [data[year]['cal']['after_cal'] for year in years])
    plt.xlabel("Year")
    plt.ylabel("AUC")
    plt.savefig("metrics/cal_over_years.pdf", format='pdf')

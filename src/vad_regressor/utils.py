from src.vad_regressor import pd, plt, os, np, ttest_ind, RobertaTokenizer, RobertaModel, ModelCheckpoint
from src.vad_regressor.config import *

def plot_vads(f, labels, preds, save=False, path=None):
    fig, axes = plt.subplots(1, 3)
    f(ax=axes[0], x=labels[:, 0], color='b')
    f(ax=axes[0], x=preds[:, 0], color='r')
    f(ax=axes[1], x=labels[:, 1], color='b')
    f(ax=axes[1], x=preds[:, 1], color='r')
    f(ax=axes[2], x=labels[:, 2], color='b')
    f(ax=axes[2], x=preds[:, 2], color='r')
    [
        (axes[i].set_xlim([MIN, MAX]),
         axes[i].set_xlabel(s))
        for i, s in enumerate(VAD_COLUMNS)
    ]

    plt.legend(['real', 'pred'])

    if save:
        plt.savefig(os.path.join(RESULTS_PATH, path))
    plt.show()


def make_hist(df, save=False, path=None):

    # means and std
    means = pd.DataFrame(df[VAD_COLUMNS].mean()).transpose()
    stds = pd.DataFrame(df[VAD_COLUMNS].std()).transpose()

    # histogram
    fig, axes = plt.subplots(1, 3)
    sns.histplot(ax=axes[0], x=df.V)
    sns.histplot(ax=axes[1], x=df.A)
    sns.histplot(ax=axes[2], x=df.D)
    [axes[i].set_xlim([MIN, MAX]) for i in range(3)]
    [axes[i].set_title(f'mean = {np.round(means[VAD_COLUMNS[i]].values[0], 3)}, '
                       f'std = {np.round(stds[VAD_COLUMNS[i]].values[0], 3)} ') for i in range(3)]
    if save:
        plt.savefig(os.path.join(RESULTS_PATH, path))
    plt.show()


def make_radar_plots(labels, preds, save=False, path=None):
    # squared error
    se = np.round((labels.to_numpy() - preds.to_numpy()) ** 2, 3)

    # Making Radar charts
    # Variables
    categories = ['Valence', 'Arousal', 'Dominance']
    N = len(categories)

    # Angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Plot
    for i in range(9):
        # Initialise the spider plot
        ax = plt.subplot(3, 3, i + 1, polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks(np.arange(MIN, MAX + 1), [str(i) for i in np.arange(MIN, MAX + 1)], color="grey",
                   size=7)
        plt.ylim(MIN, MAX)

        # set mse as title
        ax.set_title(f'SQ ER: {se[i]}')

        # Ind1
        values = preds.iloc[i]
        values = np.append(values, values[:1])
        ax.plot(angles, values, linewidth=1, linestyle='solid', label='prediction')
        ax.fill(angles, values, 'b', alpha=0.1)

        # Ind2
        values = labels.iloc[i]
        values = np.append(values, values[:1])
        ax.plot(angles, values, linewidth=1, linestyle='solid', label='real score')
        ax.fill(angles, values, 'r', alpha=0.1)

        # Ind3
        values = labels.mean().to_numpy()
        values = np.append(values, values[:1])
        ax.plot(angles, values, linewidth=1, linestyle='solid', label='mean')
        ax.fill(angles, values, 'g', alpha=0.1)

    plt.tight_layout(pad=3.0)
    plt.legend(loc='upper right', bbox_to_anchor=(0, 1.1))
    if save:
        plt.savefig(os.path.join(RESULTS_PATH, path))
        print('Radar results saved')
    plt.show()


def make_3d_plot(df, save=False, path=None):
    # 3d plot of VAD values
    plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.scatter3D(df.V, df.A, df.D)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    if save:
        plt.savefig(os.path.join(RESULTS_PATH, path))
    plt.show()


def make_scatterplot(df, save=False, path=None):
    # Subplots of variables
    fig, axs = plt.subplots(3, 1, figsize=(7, 15))

    axs[0].scatter(df.V, np.arange(df.V.shape[0]))
    axs[0].set_title('Valence')

    axs[1].scatter(df.A, np.arange(df.A.shape[0]))
    axs[1].set_title('Arousal')

    axs[2].scatter(df.D, np.arange(df.D.shape[0]))
    axs[2].set_title('Dominance')

    [axs[i].set_xlim([MIN, MAX]) for i in range(3)]

    if save:
        plt.savefig(os.path.join(RESULTS_PATH, path))
    plt.show()


def normalize_df(df, old_min, old_max, new_min, new_max):
    df.V = df.V.apply(map_range, args=(old_min, old_max, new_min, new_max))
    df.A = df.A.apply(map_range, args=(old_min, old_max, new_min, new_max))
    df.D = df.D.apply(map_range, args=(old_min, old_max, new_min, new_max))
    return df


def undersample(df, p):
    """
    :param df: Dataframe to undersample
    :param p: Percentile
    :return: Returns the dataframe, but undersampled and the indices
    """
    low_idx = get_low_perc(df, 'V', p) | get_low_perc(df, 'A', p) | get_low_perc(df, 'D', p)
    high_idx = get_high_perc(df, 'V', 1 - p) | get_high_perc(df, 'A', 1 - p) | get_high_perc(
        df, 'D', 1 - p)
    both_idx = low_idx | high_idx
    labels_all = df[both_idx]
    return labels_all, both_idx


def get_n_tails(df, n):
    largest_n = pd.concat([df.nlargest(n, 'V'), df.nlargest(n, 'A'), df.nlargest(n, 'D')])
    smallest_n = pd.concat([df.nsmallest(n, 'V'), df.nsmallest(n, 'A'), df.nsmallest(n, 'D')])
    both_n = pd.concat([largest_n, smallest_n])
    return both_n

def get_pearsonsr(df1, df2):
    return df1[VAD_COLUMNS].corrwith(df2[VAD_COLUMNS], method='pearson')


def get_ttest(df1, df2):
    return ttest_ind(df1[VAD_COLUMNS], df2[VAD_COLUMNS])

def get_stats(predictions, labels):

    # Make DataFrames
    pred_df = pd.DataFrame(predictions, columns=VAD_COLUMNS)
    labels_df = pd.DataFrame(labels, columns=VAD_COLUMNS)

    # MSE and Pearsons r
    mse = mean_squared_error(labels, predictions)
    pcorr = get_pearsonsr(pred_df, labels_df)

    # Dataframe of results
    stats = [np.array([mse, np.round(pcorr.values, 3)]).flatten()]
    stats_df = pd.DataFrame(stats, columns=['mse_V', 'mse_A', 'mse_D', 'r_V', 'r_A', 'r_D'])

    return stats_df


def get_tokenizer():
    return RobertaTokenizer.from_pretrained(MODEL_NAME)


def get_model():
    return RobertaModel.from_pretrained(MODEL_NAME)


def mean_squared_error(labels, predictions):
    return np.round(np.mean((labels - predictions) ** 2, axis=0), 3)


def get_checkpoint_callback(path):
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=path,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    return checkpoint_callback


def get_low_perc(df, col, perc):
    return df[col].le(df[col].quantile(perc))


def get_high_perc(df, col, perc):
    return df[col].ge(df[col].quantile(perc))


def map_range(val, old_min, old_max, new_min, new_max):
    return (val - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

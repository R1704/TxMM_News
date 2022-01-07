from vad_regressor import total_df, pd, plt
from vad_regressor.config import *
from vad_regressor.utils import *

# Find out how many tokens are mostly used
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
# visualize_tokens(total_df, 'text', tokenizer)

# Mean and standard deviation
means = pd.DataFrame(total_df[VAD_COLUMNS].mean()).transpose()
stds = pd.DataFrame(total_df[VAD_COLUMNS].std()).transpose()
print(means, '\n', stds)

# correlation between variables
corr_matrix = total_df[VAD_COLUMNS].corr(method='spearman')
print(corr_matrix)


def scatter_vars(x, y):
    r = round(total_df[x].corr(total_df[y], method='spearman'), 3)
    plt.scatter(total_df[[x]], total_df[[y]])
    plt.xlim([MIN, MAX])
    plt.ylim([MIN, MAX])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'Correlation: {r}')
    plt.show()

# for x, y in it.combinations(VAD_COLUMNS, 2):
#     scatter_vars(x, y)

# boxplots
# fig, ax = plt.subplots(3, 1, figsize=(7, 15))
# ax[0].boxplot(total_df.V)
# ax[1].boxplot(total_df.A)
# ax[2].boxplot(total_df.D)
# [ax[i].set_ylim([MIN, MAX]) for i in range(3)]
# [ax[i].set_xlabel(VAD_COLUMNS[i]) for i in range(3)]
# plt.show()







# make_scatterplot(total_df)
# make_3d_plot()
make_hist(total_df)

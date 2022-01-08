from src.vad_regressor import os, pd, np, pl, torch, tqdm, sns, dataframes
from src.vad_regressor.utils import *
from src.vad_regressor.config import *
from src import device, gpus
from dataset import VADDataset
from model import Regressor


# Load best model
trainer = pl.Trainer(gpus=gpus)
trained_model = Regressor.load_from_checkpoint(
    os.path.join(VAD_STORAGE, 'checkpoints/best-ckpt-roberta.ckpt')
)

# Freeze
trained_model.eval()
trained_model.freeze()

# GPU
trained_model = trained_model.to(device)

# Tokenizer
tokenizer = get_tokenizer()

# Validation Data
val_df = dataframes['dev']
val_dataset = VADDataset(
    val_df,
    tokenizer,
    max_token_len=MAX_TOKENS
)

def predict():
    # Predictions
    predictions = []
    labels = []
    # predict
    for item in tqdm(val_dataset):
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device)
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"])

    predictions = torch.stack(predictions).detach().cpu().numpy()
    labels = torch.stack(labels).detach().cpu().numpy()
    return predictions, labels




def plots(labels, predictions):
    pred_df = pd.DataFrame(predictions, columns=VAD_COLUMNS)
    labels_df = pd.DataFrame(labels, columns=VAD_COLUMNS)

    # Get the n highest and lowest of any VAD values
    tail_labels = get_n_tails(labels_df, n=15)
    tail_preds = pred_df.loc[tail_labels.index, :]

    # Get 9 random samples out of the n high/low
    tail_labels_sample = tail_labels.sample(9, random_state=42)
    tail_preds_sample = tail_preds.sample(9, random_state=42)

    # radar plot
    make_radar_plots(tail_labels_sample, tail_preds_sample, save=True, path=os.path.join(RESULTS_PATH, 'img/radar'))

    # Plot histograms and KDE results of VAD preds vs labels
    plot_vads(sns.histplot, labels, predictions, save=True, path=os.path.join(RESULTS_PATH, 'img/hist'))
    plot_vads(sns.kdeplot, labels, predictions, save=True, path=os.path.join(RESULTS_PATH, 'img/kde'))


def eval_undr(labels, predictions, blind_df):
    pred_df = pd.DataFrame(predictions, columns=VAD_COLUMNS)
    labels_df = pd.DataFrame(labels, columns=VAD_COLUMNS)

    # MSE of undersamples (highest and lowest percentile p)
    # to see how the prediction is at the tails of the distribution
    undrsmp_labl, under_idx = undersample(labels_df, p=.1)
    undrsmp_labl = undrsmp_labl.to_numpy()
    undrsmp_pred = pred_df[under_idx].to_numpy()
    mse_undr = mean_squared_error(undrsmp_labl, undrsmp_pred)
    print(f'MSE undersamples: {mse_undr}')

    # MSE tails vs blind
    undrsmp_pred = blind_df[under_idx]
    mse_undr_blind = mean_squared_error(undrsmp_labl, undrsmp_pred)
    print(f'MSE undersamples vs blind: {mse_undr_blind.to_numpy()}')



# Get predictions from model
predictions, labels = predict()
stats_pred = get_stats(predictions, labels)
print(f'Stats of Regressor')
print(stats_pred)

# Blind guess
train_df = dataframes['train']
blind = pd.concat([pd.DataFrame(train_df[VAD_COLUMNS].mean()).transpose()] * val_df.shape[0], ignore_index=True).values
stats_blind = get_stats(blind, labels)
print(f'Stats of Blind')
print(stats_blind)

ttest = get_ttest(pd.DataFrame(predictions, columns=VAD_COLUMNS), pd.DataFrame(blind, columns=VAD_COLUMNS))
print('T-test p-values of VAD between Regressor and Blind')
print(np.round(ttest[1], 4))

plots(labels, predictions)

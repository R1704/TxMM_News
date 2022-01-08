from src import os, pd, np, plt, sns, pl, torch

# ML
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Progressbar
from tqdm.auto import tqdm

# Statistics
from scipy.stats import ttest_ind

from src.vad_regressor.config import *
from src.vad_regressor.utils import *

total_df = pd.read_csv(f'{EMO_PATH}emobank.csv', index_col=0)
dataframes = {group: df for group, df in total_df.groupby('split')}

if NORMALIZE:
    for _, df in dataframes.items():
        df = normalize_df(df, OLD_MIN, OLD_MAX, NEW_MIN, NEW_MAX)

if UNDERSAMPLE:
    for _, df in dataframes.items():
        df, _ = undersample(df, p=0.1)

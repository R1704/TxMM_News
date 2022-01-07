from vad_regressor.config import *
from vad_regressor.utils import *

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
from tqdm.auto import tqdm
import seaborn as sns
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import ttest_ind
from pytorch_lightning.callbacks import ModelCheckpoint

total_df = pd.read_csv(f'{EMO_PATH}emobank.csv', index_col=0)
dataframes = {group: df for group, df in total_df.groupby('split')}

if NORMALIZE:
    for _, df in dataframes.items():
        df = normalize_df(df, OLD_MIN, OLD_MAX, NEW_MIN, NEW_MAX)

if UNDERSAMPLE:
    for _, df in dataframes.items():
        df, _ = undersample(df, p=0.1)

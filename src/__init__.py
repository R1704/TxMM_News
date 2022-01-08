# Paths
import os

# ML
import torch
import pytorch_lightning as pl
from transformers import RobertaTokenizer

# Data handling
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Language
import nltk
from nltk import sent_tokenize

from src.vad_regressor.model import Regressor
from src.vad_regressor.config import *

# nltk.download('punkt')

# # GPU settings
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# gpus = 1 if torch.cuda.is_available() else 0
#
# # Get model
# trainer = pl.Trainer(gpus=gpus)
# trained_model = Regressor.load_from_checkpoint(VAD_CKPT_PATH)
# trained_model = trained_model.to(device)
#
# # Freeze
# trained_model.eval()
# trained_model.freeze()
#
# # Get tokenizer
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

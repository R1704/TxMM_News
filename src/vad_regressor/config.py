from src.config import *


EMO_PATH        = os.path.join(DATASETS_PATH, 'EmoBank/corpus/')
VAD_STORAGE     = os.path.join(MODELS_PATH, 'vadRegressor/')
VAD_CKPT_PATH   = os.path.join(VAD_STORAGE, 'checkpoints/best-ckpt-roberta.ckpt')

MODEL_NAME      = 'roberta-large'
MAX_TOKENS      = 128
NUM_WORKERS     = 20
BATCH_SIZE      = 8
N_EPOCHS        = 10

VAD_COLUMNS     = ['V', 'A', 'D']
NORMALIZE       = False
UNDERSAMPLE     = False
OLD_MIN         = 1
OLD_MAX         = 5
NEW_MIN         = 0
NEW_MAX         = 1
if NORMALIZE:
    MIN = NEW_MIN
    MAX = NEW_MAX
else:
    MIN = OLD_MIN
    MAX = OLD_MAX

# one model for all VAD values or one model for each (should be 1 or 3)
N_OUTPUTS = 3


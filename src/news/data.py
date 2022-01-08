from src import pd, plt, np, os
from src.config import *
from src.vad_regressor.config import *
from src.news.utils import *


# Clean data
abcnews_path = os.path.join(TxMM_PATH, 'abcnews-date-text.csv')
df = pd.read_csv(abcnews_path)
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')

# Compute headline length
df['length'] = df.headline_text.str.split().str.len()

# Apply VAD regressor on all data
df[VAD_COLUMNS] = df.headline_text.apply(predict_vad)
df.to_csv('vad_preds.csv')

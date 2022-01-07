from src import pd, plt, np
from src.config import *
from vad_regressor.config import *
from news.utils import *

# Clean data
df = pd.read_csv(DATA_PATH)
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')

# Compute headline length
df['length'] = df.headline_text.str.split().str.len()

# Apply VAD regressor on all data
df[VAD_COLUMNS] = df.headline_text.apply(predict_vad)

plt.plot(np.arange(df.shape[0]), df['V'])
plt.show()

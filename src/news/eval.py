from src import os, pd, plt, np
from src.config import *

news_preds_path = os.path.join(TxMM_PATH, 'vad_preds.csv')
df = pd.read_csv(news_preds_path)
print(df.head())

df['publish_date'] = pd.to_datetime(df.publish_date)
# plt.plot(np.arange(df.shape[0]), df.V)
grouped = df.groupby(df.publish_date.dt.year)

plt.plot(grouped['A'].mean())
plt.ylim([1, 5])
# plt.xticks(np.linspace(2003, 2021, 9), rotation=90)
plt.show()

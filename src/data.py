from src import pd, plt
from src.config import *

# Clean data
data = pd.read_csv(DATA_PATH)
data['publish_date'] = pd.to_datetime(data['publish_date'], format='%Y%m%d')

# Apply VAD regressor on all data ok.

# data['length'] = data.headline_text.map(len)
data['length'] = data.headline_text.str.split().str.len()
# plt.scatter(data.publish_date, data['length'])
grouped = data.groupby('publish_date')['length'].mean()
print(grouped.head())
# plt.scatter(data.groupby(['publish_date']), data.groupby(['publish_date'])['length'])
# plt.scatter(grouped, grouped.mean())
plt.plot(grouped)
plt.show()

print(data, data.shape)


# if __name__ == '__main__':
#     pass

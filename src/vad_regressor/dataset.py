from src.vad_regressor import torch, pd, Dataset
from src.vad_regressor.config import *


class VADDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer,
            max_token_len: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row.text
        labels = row[VAD_COLUMNS]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_TOKENS,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        as_dict = dict(
            text=text,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.FloatTensor(labels)
        )
        return as_dict

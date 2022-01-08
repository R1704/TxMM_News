from src.vad_regressor import pl, DataLoader
from src.vad_regressor.config import *
from dataset import VADDataset


class VADModule(pl.LightningDataModule):

    def __init__(self, train_df, test_df, val_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = VADDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = VADDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

        self.val_dataset = VADDataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS
        )

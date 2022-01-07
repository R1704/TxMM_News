# Imports
from vad_regressor import os, pl, TensorBoardLogger, EarlyStopping
from module import VADModule
from model import Regressor
from vad_regressor import dataframes
from vad_regressor.utils import *


# Tokenizer
tokenizer = get_tokenizer()

# Data module
data_module = VADModule(
    train_df=dataframes['train'],
    test_df=dataframes['test'],
    val_df=dataframes['dev'],
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKENS
)

# variables for warmup
steps_per_epoch = len(dataframes['train']) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5

# Model
model = Regressor(
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps
)

# Save and log
checkpoint_callback = get_checkpoint_callback(path=VAD_CKPT_PATH)
logger = TensorBoardLogger(os.path.join(VAD_STORAGE, 'lightning_logs'), name='texts')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

# Trainer
trainer = pl.Trainer(
    logger=logger,
    callbacks=[early_stopping_callback, checkpoint_callback],
    max_epochs=N_EPOCHS,
    gpus=1,  # use str to specify gpu, e.g. '1' for GPU 1
    progress_bar_refresh_rate=30
)

# Train the model
trainer.fit(model, data_module)
print(trainer.test())

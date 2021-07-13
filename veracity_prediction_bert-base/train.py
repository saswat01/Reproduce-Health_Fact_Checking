import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import torch
# from torch.utils.data import DataLoader
import config
from dataset import veracityModule, inputData
from model import veracityPreds

def train():
    train_data = pd.read_csv(config.TRAINING_FILE)
    val_data = pd.read_csv(config.VALIDATION_FILE)
    test_data = pd.read_csv(config.TESTING_FILE)

    train_data['label'] = train_data.label.apply(to_sentiment)
    val_data['label'] = val_data.label.apply(to_sentiment)
    test_data['label'] = test_data.label.apply(to_sentiment)
    data_module = veracityModule(train_data, test_data, val_data, config.BATCH_SIZE)

    #device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    #loss_fn = torch.nn.CrossEntropyLoss().to(device)
    #num_training_steps = config.N_EPOCHS * len(DataLoader(inputData(train_data)))
    steps_per_epoch=len(train_data) // config.BATCH_SIZE
    total_training_steps = steps_per_epoch * config.N_EPOCHS
    warmup_steps = total_training_steps // 5

    checkpoint_callback = ModelCheckpoint(
    dirpath=config.CHECKPOINT_PATH,
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
    ) 

    bert_model = veracityPreds(config.N_CLASSES, n_warmup_steps=warmup_steps, n_training_steps=total_training_steps )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)  

    trainer = pl.Trainer(
    checkpoint_callback=checkpoint_callback,
    callbacks=[checkpoint_callback, early_stopping_callback], 
    #gpus=1, #use this only if you have gpu 
    max_epochs=config.N_EPOCHS, 
    progress_bar_refresh_rate=30 
    )
    print("-"*100)
    print("Start training")
    print('-'*100)
    trainer.fit(bert_model, data_module) 
    trainer.test(bert_model)

def to_sentiment(rating):
    rating = str(rating)
    if rating == 'false':
        return 0
    elif rating == 'mixture':
        return 2
    elif rating == 'true': 
        return 1
    else:
        return 3

if __name__=="__main__":
    train()
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import torch
# from torch.utils.data import DataLoader
from dataset import veracityModule, inputData
from model import veracityPreds
import configparser
import argparse

def train(args):
    train_data = pd.read_csv(args.train_path)
    val_data = pd.read_csv(args.val_path)
    test_data = pd.read_csv(args.test_path)

    train_data['label'] = train_data.label.apply(to_sentiment)
    val_data['label'] = val_data.label.apply(to_sentiment)
    test_data['label'] = test_data.label.apply(to_sentiment)
    data_module = veracityModule(train_data, test_data, val_data, args.batch_size)

    #device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    #loss_fn = torch.nn.CrossEntropyLoss().to(device)
    #num_training_steps = config.N_EPOCHS * len(DataLoader(inputData(train_data)))
    steps_per_epoch=len(train_data) // args.batch_size
    total_training_steps = steps_per_epoch * args.n_epochs
    warmup_steps = total_training_steps // 5

    checkpoint_callback = ModelCheckpoint(
    dirpath=args.checkpoint_path,
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
    ) 

    bert_model = veracityPreds(args.n_classes, n_warmup_steps=warmup_steps, n_training_steps=total_training_steps )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)  

    trainer = pl.Trainer(
    checkpoint_callback=checkpoint_callback,
    callbacks=[checkpoint_callback, early_stopping_callback], 
    #gpus=1, #use this only if you have gpu 
    max_epochs=args.n_epochs, 
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

def _parse_args():
    """Parse command-line arguments"""
    config = fetch_config('config.ini')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default=config['Paths']['TRAINING_FILE'])
    parser.add_argument('--val_path', default=config['Paths']['VALIDATION_FILE'])
    parser.add_argument('--test_path', default=config['Paths']['TESTING_FILE'])
    parser.add_argument("--n_epochs", type=int, default=int(config['Model']['N_EPOCHS']))
    parser.add_argument('--max_len', type=int, default=int(config['Model']['MAX_LEN']))
    parser.add_argument('--batch_size', type=int, default=int(config['Model']['BATCH_SIZE']))
    parser.add_argument('-lr', type=float, default=float(config['Model']['LEARNING_PATH']))
    parser.add_argument('--n_classes', type=int, default=int(config['Model']['N_CLASSES']))
    parser.add_argument("--checkpoint_path", default=config['Model']['CHECKPOINT_PATH'])
    args = parser.parse_args()
    return args

def fetch_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config

def main():
    args = _parse_args()
    train(args)

if __name__=="__main__":
    main()
    
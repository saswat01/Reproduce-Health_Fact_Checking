import config
import model
import dataset
import pytorch_lightning as pl
from pytorch_lightning import ModelCheckpoint
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

def metric_show():
    checkpoint_dir = config.CHECKPOINT_PATH
    scibert_model = model.veracityPreds.load_from_checkpoint(checkpoint_dir, config.N_CLASSES)
    scibert_model.eval()
    scibert_model.freeze()

    val_dataset = pd.read_csv(config.TESTING_FILE)
    val_dataset['label'] = val_dataset.label.apply(to_sentiment)

    from tqdm.auto import tqdm
    device = 'cpu'
    trained_model = scibert_model.to(device)
    val_dataset = dataset.inputData()

    predictions = []
    labels = []

    for item in tqdm(val_dataset):
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device), 
            item["attention_mask"].unsqueeze(dim=0).to(device)
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())

        predictions = torch.stack(predictions).detach().cpu()
        labels = torch.stack(labels).detach().cpu()

    print(classification_report(labels, torch.argmax(F.softmax(predictions, dim=1), dim=1)))
    

    arr = confusion_matrix(labels, torch.argmax(F.softmax(predictions, dim=1), dim=1))
    plt.figure(figsize = (10,7))
    sns.heatmap(arr/np.sum(arr), annot=True, cmap='Blues')

if __name__=="__main__":
    metric_show()